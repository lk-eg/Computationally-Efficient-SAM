import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class VASSOREMUCRT(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, logger, rho, theta, k) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"

        self.base_optimizer = base_optimizer
        self.logger = logger

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert theta <= 1 and theta >= 0, f"theta must live in [0, 1]."
        self.rho = rho
        self.theta = theta
        self.k = k

        self.tau = 0
        self.iteration_step_counter = 0
        self.inner_grad_calc_counter = 0
        
        super(VASSOREMUCRT, self).__init__(params, dict(rho=rho, theta=theta))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group['rho'] = rho
            group['theta'] = theta

            for p in group['params']:
                self.state[p]['e'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['g_t'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['g_{t-1}'] = torch.zeros_like(p, requires_grad=False).to(p)

        custom_metrics_per_batch = ['inner_grad_calc_counter']
        self.logger.wandb_define_metrics_per_batch(custom_metrics_per_batch)

    @classmethod
    def from_config(cls, args):
        return {
            'rho': args.rho,
            'theta': args.theta,
            'k': args.k,
        }
    
    def crt(self):
        if self.iteration_step_counter > 100:
            prev_g_norm = self._avg_grad_norm('g_t')
            return self.tau > prev_g_norm
    
    @torch.enable_grad()
    def inner_gradient_calculation(self, crt, model, images, targets, criterion):
        if not crt:
            self.inner_grad_calc_counter += 1
            self.logger.wandb_log_batch(**{'inner_grad_calc_counter': self.inner_grad_calc_counter, 'global_batch_counter': self.iteration_step_counter})
            output = model(images)
            loss = criterion(output, targets)
            self.base_optimizer.zero_grad()
            loss.backward()
            loss_w_t = loss.item()


    @torch.no_grad()
    def first_step(self, crt, zero_grad=False):
        for group in self.param_groups:
            theta = group['theta']
            for p in group['params']:
                if p.grad is None: continue
                if 'ema' not in self.state[p]:
                    self.state[p]['ema'] = p.grad.clone().detach()
                else:
                    self.state[p]['ema'].mul_(1 - theta)
                    gradient = torch.zeros_like(p).to(p)
                    if not crt:
                        gradient = p.grad
                    else:
                        gradient = self.state[p]['g_{t-1}']
                    self.state[p]['ema'].add_(gradient, alpha=theta)

        avg_grad_norm = self._avg_grad_norm('ema')
        for group in self.param_groups:
            scale = group['rho'] / (avg_grad_norm + 1e-7)

            for p in group['params']:
                e_w = self.state[p]['ema'] * scale
                self.state[p]['e'] = e_w.clone()
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]['e'])

                self.state[p]['g_{t-1}'] = self.state[p]['g_t']
                self.state[p]['g_t'] = p.grad.clone()

        g_norm = self._avg_grad_norm('g_t')
        self.tau = (1 - self.theta) * self.tau + self.theta * g_norm

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        model = kwargs['model']
        images = kwargs['images']
        targets = kwargs['targets']
        criterion = kwargs['criterion']

        crt = self.crt()

        self.inner_gradient_calculation(crt, model, images, targets, criterion)
        self.first_step(crt, True)
        with torch.enable_grad():
            output, loss = closure()
        self.second_step()

        self.iteration_step_counter += 1

        return output, loss


    def _avg_grad_norm(self, key):
        norm = torch.norm(
            torch.stack([
                self.state[p][key].norm(p=2)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm