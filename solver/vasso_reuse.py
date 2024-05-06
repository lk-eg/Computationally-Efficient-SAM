import math
import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class VASSORE(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, logger, rho, theta, k, reuse_random_perturbation) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        self.logger = logger

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert theta <= 1 and theta >= 0, f"theta must live in [0, 1]."
        self.rho = rho
        self.theta = theta
        self.k = k
        self.randomPerturbationComponent = reuse_random_perturbation
        self.iteration_step_counter = 0
        self.cos_sim = 0
        super(VASSORE, self).__init__(params, dict(rho=rho, theta=theta))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group['rho'] = rho
            group['theta'] = theta

            for p in group['params']:
                self.state[p]['e'] = torch.zeros_like(p, requires_grad=False).to(p)

    @classmethod
    def from_config(cls, args):
        return {
            'rho': args.rho,
            'theta': args.theta,
            'k': args.k,
            'reuse_random_perturbation': args.reuse_random_perturbation
        }

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        shared_device = self.param_groups[0]["params"][0].device
        if not self.iteration_step_counter % self.k:
            for group in self.param_groups:
                theta = group['theta']
                for p in group['params']:
                    if p.grad is None: continue
                    if 'ema' not in self.state[p]:
                        self.state[p]['ema'] = p.grad.clone().detach()
                    else:
                        self.state[p]['ema'].mul_(1 - theta)
                        self.state[p]['ema'].add_(p.grad, alpha=theta)

        for group in self.param_groups:
            if not self.iteration_step_counter % self.k:
                avg_grad_norm = self._avg_grad_norm()
                scale = group["rho"] / (avg_grad_norm + 1e-7)

            for p in group["params"]:
                if not self.iteration_step_counter % self.k:
                    if p.grad is None: continue
                    e_w = self.state[p]['ema'] * scale
                    self.state[p]['e'] = e_w.clone()
                elif self.randomPerturbationComponent:
                    e_size = self.state[p]['e'].size()
                    uniform_noise = torch.rand(e_size).to(shared_device) * 1e-3
                    e_w = self.state[p]['e'] + uniform_noise
                else:
                    e_w = self.state[p]['e']

                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]['e'])
        
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

    def _avg_grad_norm(self):
        norm = torch.norm(
            torch.stack([
                self.state[p]['ema'].norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm