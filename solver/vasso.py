import math
import torch
import torch.optim
import numpy as np

from utils.configurable import configurable
from scipy.stats import spearmanr, pearsonr

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class VASSO(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, logger, rho, theta, momentum) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        self.logger = logger

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert theta <= 1 and theta >= 0, f"theta must live in [0, 1]."
        self.rho = rho
        self.theta = theta
        self.momentum = momentum
        self.iteration_step_counter = 0
        self.normdiff = 0
        self.cos_sim = 0

        super(VASSO, self).__init__(params, dict(rho=rho, theta=theta))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group['theta'] = theta

            for p in group['params']:
                self.state[p]['e_t'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['e_{t-1}'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['w_t'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['w_{t-1}'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['grad'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['b_t'] = torch.zeros_like(p, requires_grad=False)
        self.cos_sim_evolution = []
        self.w_t_normdiff_evolution = []
        self.b_t_norm_evolution = []

        # define here the custom metrics that will be tracked per batch
        custom_metrics = ['cosSim(e_t, e_{t-1})', '||w_t - w_{t-1}||']
        self.logger.wandb_define_metrics_per_batch(custom_metrics)

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho,
            'theta': args.theta,
            'momentum': args.momentum, # only for sgd. If I want to make it more general, I will have to remove this at some point. Or maybe I don't have to remove it.
        }

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            theta = group['theta']
            for p in group["params"]:
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
        
        self.normdiff = self._normdiff('w_t', 'w_{t-1}')
        self.cos_sim = self._cosine_similarity('e_t', 'e_{t-1}')

        # this here is not scaled by learning rate
        # self.w_t_normdiff_evolution.append(self.normdiff.item())
        self.cos_sim_evolution.append(self.cos_sim)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_t'])
                self.state[p]['grad'] = p.grad.clone().detach()

                if 'momentum_buffer' in self.base_optimizer.state[p]:
                    momentum_buffer = self.base_optimizer.state[p]['momentum_buffer']
                    self.state[p]['b_t'] = self.momentum * momentum_buffer + self.state[p]['grad']

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        logger = kwargs['logger']
        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

        self.iteration_step_counter += 1
        self.logger.wandb_log_batch(**{'||w_t - w_{t-1}||': self.normdiff, 'global_batch_counter': self.iteration_step_counter})
        self.logger.wandb_log_batch(**{'cosSim(e_t, e_{t-1})': self.cos_sim, 'global_batch_counter': self.iteration_step_counter})
        current_gradient_norm = self._avg_grad_norm('grad')
        self.logger.wandb_log_batch(**{'||g_t||': current_gradient_norm, 'global_batch_counter': self.iteration_step_counter})
        self.w_t_normdiff_evolution.append(current_gradient_norm)
        current_b_t_norm = self._avg_grad_norm('b_t')
        self.logger.wandb_log_batch(**{'||b_t||': current_b_t_norm, 'global_batch_counter': self.iteration_step_counter})
        self.b_t_norm_evolution.append(current_b_t_norm)

        if self.iteration_step_counter % 100 == 0:
            if self.iteration_step_counter == 100:
                self.cos_sim_evolution.pop(0)
                self.w_t_normdiff_evolution.pop(0)
                self.b_t_norm_evolution.pop(0)

            et_values_arr = np.array(self.cos_sim_evolution)
            wt_norm_values_arr = np.array(self.w_t_normdiff_evolution)
            b_t_norm_values_arr = np.array(self.b_t_norm_evolution)
            pearson_corr_1, _ = pearsonr(et_values_arr, wt_norm_values_arr)
            self.logger.log(f'=====*****===== PEARSON_CORR_1=  {pearson_corr_1}')
            self.logger.wandb_log_batch(**{'PEARSON_CORR_CUM(||g_t||, cosSim)': pearson_corr_1, 'global_batch_counter': self.iteration_step_counter})
            pearson_corr_2, _ = pearsonr(et_values_arr, b_t_norm_values_arr)
            self.logger.log(f'=====*****===== PEARSON_CORR_2=  {pearson_corr_2}')
            self.logger.wandb_log_batch(**{'PEARSON_CORR_CUM(||b_t||, cosSim)': pearson_corr_2, 'global_batch_counter': self.iteration_step_counter})

            spearman_corr_1, _ = spearmanr(et_values_arr, wt_norm_values_arr)
            self.logger.log(f'=====*****===== SPEARMAN_CORR_1=  {spearman_corr_1}')
            self.logger.wandb_log_batch(**{'SPEARMAN_CORR_CUM(||g_t||, cosSim)': spearman_corr_1, 'global_batch_counter': self.iteration_step_counter})
            spearman_corr_2, _ = spearmanr(et_values_arr, b_t_norm_values_arr)
            self.logger.log(f'=====*****===== SPEARMAN_CORR_2=  {spearman_corr_2}')
            self.logger.wandb_log_batch(**{'SPEARMAN_CORR_CUM(|b_t||, cosSim)': spearman_corr_2, 'global_batch_counter': self.iteration_step_counter})


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