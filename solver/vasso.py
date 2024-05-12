import torch
import torch.optim
import numpy as np

from utils.configurable import configurable
from scipy.stats import spearmanr, pearsonr

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class VASSO(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, logger, rho, theta, momentum, max_epochs) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        self.logger = logger

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert theta <= 1 and theta >= 0, f"theta must live in [0, 1]."
        self.rho = rho
        self.theta = theta
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.iteration_step_counter = 0
        self.normdiff = 0
        self.cos_sim = 0

        super(VASSO, self).__init__(params, dict(rho=rho, theta=theta))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group['rho'] = rho
            group['theta'] = theta

            for p in group['params']:
                itr_metric_keys = ['e_t', 'e_{t-1}', 'w_t', 'w_{t-1}', 'g_t', 'g_{t-1}', 'pert_t', 'pert_{t-1}']
                for key in itr_metric_keys:
                    self.state[p][key] = torch.zeros_like(p, requires_grad=False).to(p)

        self.cos_sim_evolution_all_epochs = []
        self.cos_sim_evolution_training_stage = []

        self.w_normdiff_evolution_all_epochs = []
        self.w_normdiff_evolution_training_stage = []

        self.pert_normdiff_evolution_all_epochs = []
        self.pert_normdiff_evolution_training_stage = []

        self.g_prev_norm_evolution_all_epochs = []
        self.g_prev_norm_evolution_training_stage = []

        # define here the custom metrics that will be tracked per batch
        custom_metrics_per_batch = ['cosSim(e_t, e_{t-1})', '||w_t - w_{t-1}||', '||g_{t-1}||', '||pert_t - pert_{t-1}||']
        self.logger.wandb_define_metrics_per_batch(custom_metrics_per_batch)

        # define here the custom metrics that will be tracked per training stage
        custom_metrics_per_training_stage = ['PEARSON_CORR_STAGE(||g_{t-1}||, cosSim)', 'SPEARMAN_CORR_STAGE(||g_{t-1}||, cosSim)', 
                                             'p-value_||g_{t-1}||', 'q-value_||g_{t-1}||',
                                             'PEARSON_CORR_STAGE(||w_t - w_{t-1}||, cosSim)', 'SPEARMAN_CORR_STAGE(||w_t - w_{t-1}||, cosSim)', 
                                             'r-value_||w_t - w_{t-1}||', 's-value_||w_t - w_{t-1}||'
                                             ]
        self.logger.wandb_define_metrics_per_training_stage(custom_metrics_per_training_stage)


    @classmethod
    def from_config(cls, args):
        return {
            'rho': args.rho,
            'theta': args.theta,
            'momentum': args.momentum, # only for sgd. If I want to make it more general, I will have to remove this at some point. Or maybe I don't have to remove it.
            'max_epochs': args.epochs
        }
    
    @torch.enable_grad()
    def inner_gradient_calculation(self, model, images, targets, criterion):
        output = model(images)
        loss = criterion(output, targets)
        self.base_optimizer.zero_grad()
        loss.backward()
        loss_w_t = loss.item()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            theta = group['theta']
            for p in group['params']:
                if p.grad is None: continue
                if 'ema' not in self.state[p]:
                    self.state[p]['ema'] = p.grad.clone().detach()
                else:
                    self.state[p]['ema'].mul_(1 - theta)
                    self.state[p]['ema'].add_(p.grad, alpha=theta)

                self.state[p]['w_{t-1}'] = self.state[p]['w_t'].clone()
                self.state[p]['w_t'] = p.clone().detach()

        avg_grad_norm = self._avg_grad_norm('ema')
        for group in self.param_groups:
            scale = group["rho"] / (avg_grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = self.state[p]['ema'] * scale
                p.add_(e_w)

                self.state[p]['e_{t-1}'] = self.state[p]['e_t'].clone()
                self.state[p]['e_t'] = e_w.clone()

                self.state[p]['pert_{t-1}'] = self.state[p]['pert_t'].clone()
                self.state[p]['pert_t'] = p.clone().detach()
        
        self.normdiff = self._normdiff('w_t', 'w_{t-1}')
        self.pert_normdiff = self._normdiff('pert_t', 'pert_{t-1}')
        self.cos_sim = self._cosine_similarity('e_t', 'e_{t-1}')

        # update the lists that will be used for measuring correlations
        if not self.iteration_step_counter == 0:
            self.cos_sim_evolution_all_epochs.append(self.cos_sim)
            self.cos_sim_evolution_training_stage.append(self.cos_sim)

            self.w_normdiff_evolution_all_epochs.append(self.normdiff.item())
            self.w_normdiff_evolution_training_stage.append(self.normdiff.item())

            self.pert_normdiff_evolution_all_epochs.append(self.pert_normdiff.item())
            self.pert_normdiff_evolution_training_stage.append(self.pert_normdiff.item())

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_t'])

                # I am running here an analysis on the outer gradient, g_{SAM}, not the inner gradient.
                self.state[p]['g_{t-1}'] = self.state[p]['g_t'].clone()
                self.state[p]['g_t'] = p.grad.clone().detach()

                # if 'momentum_buffer' in self.base_optimizer.state[p]:
                #     momentum_buffer = self.base_optimizer.state[p]['momentum_buffer']
                #     self.state[p]['b_t'] = self.momentum * momentum_buffer + self.state[p]['grad']

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        epoch = kwargs['epoch']
        model = kwargs['model']
        images = kwargs['images']
        targets = kwargs['targets']
        criterion = kwargs['criterion']

        self.inner_gradient_calculation(model, images, targets, criterion)
        self.first_step(True)
        with torch.enable_grad():
            output, loss = closure()
        self.second_step()

        self.iteration_step_counter += 1

        # logging I am interested in
        self._metrics_logging()
        self._correlation_logging(epoch) 

        return output, loss

            
    
    def _avg_grad_norm(self, key):
        norm = torch.norm(
            torch.stack([
                self.state[p][key].norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def _normdiff(self, key1, key2):
        shared_device = self.param_groups[0]["params"][0].device
        vectors = []
        for group in self.param_groups:
            for p in group["params"]:
                diff = torch.sub(self.state[p][key1], self.state[p][key2])
                vectors.append(diff.norm(p=2).to(shared_device))

        norm = torch.norm(torch.stack(vectors), p=2)
        return norm
    
    def _cosine_similarity(self, key1, key2):
        flattened_tensors1 = []
        flattened_tensors2 = []
        for group in self.param_groups:
            for p in group["params"]:
                flattened_tensors1.append(torch.flatten(self.state[p][key1]))
                flattened_tensors2.append(torch.flatten(self.state[p][key2]))

        concatenated_tensor1 = torch.cat(flattened_tensors1, dim=0)
        concatenated_tensor2 = torch.cat(flattened_tensors2, dim=0)
        dot_product = torch.dot(concatenated_tensor1, concatenated_tensor2)

        norm_a = torch.norm(concatenated_tensor1, p=2)
        norm_b = torch.norm(concatenated_tensor2, p=2)
        cosine_similarity = dot_product/(norm_a * norm_b)
        
        return cosine_similarity.item()
    
    def _metrics_logging(self):
        self.logger.wandb_log_batch(**{'||w_t - w_{t-1}||': self.normdiff.item(), 'global_batch_counter': self.iteration_step_counter})
        self.logger.wandb_log_batch(**{'cosSim(e_t, e_{t-1})': self.cos_sim, 'global_batch_counter': self.iteration_step_counter})
        self.logger.wandb_log_batch(**{'||pert_t - pert_{t-1}||': self.pert_normdiff.item(), 'global_batch_counter': self.iteration_step_counter})

        previous_sam_gradient_norm = self._avg_grad_norm('g_{t-1}').item()
        self.logger.wandb_log_batch(**{'||g_{t-1}||': previous_sam_gradient_norm, 'global_batch_counter': self.iteration_step_counter})
        if not self.iteration_step_counter == 1:
            self.g_prev_norm_evolution_all_epochs.append(previous_sam_gradient_norm)
            self.g_prev_norm_evolution_training_stage.append(previous_sam_gradient_norm)

    def _correlation_logging(self, epoch):
        if epoch % 5 == 1 and not self.logged_epoch == epoch:
            self.logged_epoch = epoch

            cos_sim_values_arr = np.array(self.cos_sim_evolution_training_stage)
            g_prev_arr = np.array(self.g_prev_norm_evolution_training_stage)
            w_normdiff_arr = np.array(self.w_normdiff_evolution_training_stage)
            pert_normdiff_arr = np.array(self.pert_normdiff_evolution_training_stage)

            pearson_corr_g_prev_cos_sim, p1 = pearsonr(g_prev_arr, cos_sim_values_arr)
            self.logger.wandb_log_batch(**{'PEARSON_CORR_STAGE(||g_{t-1}||, cosSim)': pearson_corr_g_prev_cos_sim, 'p-value_||g_{t-1}||': p1, 'training_stage_%5': epoch//5})

            spearman_corr_g_prev_cos_sim, q1 = spearmanr(g_prev_arr, cos_sim_values_arr)
            self.logger.wandb_log_batch(**{'SPEARMAN_CORR_STAGE(||g_{t-1}||, cosSim)': spearman_corr_g_prev_cos_sim, 'q-value_||g_{t-1}||': q1, 'training_stage_%5': epoch//5})

            pearson_corr_w_normdiff_cos_sim, r1 = pearsonr(w_normdiff_arr, cos_sim_values_arr)
            self.logger.wandb_log_batch(**{'PEARSON_CORR_STAGE(||w_t - w_{t-1}||, cosSim)': pearson_corr_w_normdiff_cos_sim, 'r-value_||w_t - w_{t-1}||': r1, 'training_stage_%5': epoch//5})

            spearman_corr_w_normdiff_cos_sim, s1 = pearsonr(w_normdiff_arr, cos_sim_values_arr)
            self.logger.wandb_log_batch(**{'SPEARMAN_CORR_STAGE(||w_t - w_{t-1}||, cosSim)': spearman_corr_w_normdiff_cos_sim, 's-value_||w_t - w_{t-1}||': s1, 'training_stage_%5': epoch//5})

            pearson_corr_pert_normdiff_cos_sim, pp1 = pearsonr(pert_normdiff_arr, cos_sim_values_arr)
            self.logger.wandb_log_batch(**{'PEARSON_CORR_STAGE(||pert_t - pert_{t-1}||, cosSim)': pearson_corr_pert_normdiff_cos_sim, 'pp-value_||pert_t - pert_{t-1}||': pp1, 'training_stage_%5': epoch//5})

            pearson_corr_pert_normdiff_g_prev, qq1 = pearsonr(pert_normdiff_arr, g_prev_arr)
            self.logger.wandb_log_batch(**{'PEARSON_CORR_STAGE(||pert_t - pert_{t-1}||, ||g_{t-1}||)': pearson_corr_pert_normdiff_g_prev, 'qq-value_||pert_t - pert_{t-1}||': qq1, 'training_stage_%5': epoch//5})

            pearson_corr_pert_normdiff_w_normdiff, rr1 = pearsonr(pert_normdiff_arr, w_normdiff_arr)
            self.logger.wandb_log_batch(**{'PEARSON_CORR_STAGE(||pert_t - pert_{t-1}||, ||w_t - w_{t-1}||)': pearson_corr_pert_normdiff_w_normdiff, 'rr-value_||pert_t - pert_{t-1}||': rr1, 'training_stage_%5': epoch//5})

            self.cos_sim_evolution_training_stage.clear()
            self.g_prev_norm_evolution_training_stage.clear()
            self.w_normdiff_evolution_training_stage.clear()
            self.pert_normdiff_evolution_training_stage.clear()

        if epoch == self.max_epochs-1:
            cos_sim_values_all_epochs_arr = np.array(self.cos_sim_evolution_all_epochs)
            g_prev_all_epochs_arr = np.array(self.g_prev_norm_evolution_all_epochs)
            w_normdiff_all_epochs_arr = np.array(self.w_normdiff_evolution_all_epochs)
            pert_normdiff_all_epochs_arr = np.array(self.pert_normdiff_evolution_all_epochs)

            pearson_corr_g_prev_cos_sim_all_epochs, p2 = pearsonr(g_prev_all_epochs_arr, cos_sim_values_all_epochs_arr)
            self.logger.log(f'=====*****===== PEARSON_CORR_GLOBAL(||g_{{t-1}}||, cosSim) =  {pearson_corr_g_prev_cos_sim_all_epochs}, p-value_||g_{{t-1}}||: {p2}')
            spearman_corr_g_prev_cos_sim_all_epochs, q2 = spearmanr(g_prev_all_epochs_arr, cos_sim_values_all_epochs_arr)
            self.logger.log(f'=====*****===== SPEARMAN_CORR_GLOBAL(||g_{{t-1}}||, cosSim) =  {spearman_corr_g_prev_cos_sim_all_epochs}, q-value_||g_{{t-1}}||: {q2}')

            pearson_corr_w_normdiff_cos_sim_all_epochs, r2 = pearsonr(w_normdiff_all_epochs_arr, cos_sim_values_all_epochs_arr)
            self.logger.log(f'=====*****===== PEARSON_CORR_GLOBAL(||w_t - w_{{t-1}}||, cosSim) =  {pearson_corr_w_normdiff_cos_sim_all_epochs}, r-value_||w_t - w_{{t-1}}||: {r2}')
            spearman_corr_w_normdiff_cos_sim_all_epochs, s2 = spearmanr(w_normdiff_all_epochs_arr, cos_sim_values_all_epochs_arr)
            self.logger.log(f'=====*****===== SPEARMAN_CORR_GLOBAL(||w_t - w_{{t-1}}||, cosSim) =  {spearman_corr_w_normdiff_cos_sim_all_epochs}, s-value_||w_t - w_{{t-1}}||: {s2}')

            pearson_corr_pert_normdiff_cos_sim_all_epochs, pp2 = pearsonr(pert_normdiff_all_epochs_arr, cos_sim_values_all_epochs_arr)
            self.logger.log(f'=====*****===== PEARSON_CORR_GLOBAL(||pert_t - pert_{{t-1}}||, cosSim) =  {pearson_corr_pert_normdiff_cos_sim_all_epochs}, p-value_||pert_t - pert_{{t-1}}||: {pp2}')
            spearman_corr_pert_normdiff_cos_sim_all_epochs, qq2 = spearmanr(pert_normdiff_all_epochs_arr, cos_sim_values_all_epochs_arr)
            self.logger.log(f'=====*****===== SPEARMAN_CORR_GLOBAL(||pert_t - pert_{{t-1}}||, cosSim) =  {spearman_corr_pert_normdiff_cos_sim_all_epochs}, q-value_||pert_t - pert_{{t-1}}||: {qq2}')
