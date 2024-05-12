import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class VASSOREMU(torch.optim.Optimizer):
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

        self.iteration_step_counter = 0

        super(VASSOREMU, self).__init__(params, dict(rho=rho, theta=theta))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group['rho'] = rho
            group['theta'] = theta

            for p in group['params']:
                self.state[p]['e'] = torch.zeros_like(p, requires_grad=False).to(p)
                self.state[p]['g_{t-1}'] = torch.zeros_like(p, requires_grad=False).to(p)

    @classmethod
    def from_config(cls, args):
        return {
            'rho': args.rho,
            'theta': args.theta,
            'k': args.k
        }
    
    @torch.enable_grad()
    def inner_gradient_calculation(self, model, images, targets, criterion):
        if self.iteration_step_counter % self.k == 0:
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
                    gradient = torch.zeros_like(p).to(p)
                    if self.iteration_step_counter % self.k == 0:
                        gradient = p.grad
                    else:
                        gradient = self.state[p]['g_{t-1}']
                    self.state[p]['ema'].add_(gradient, alpha=theta)

        for group in self.param_groups:
            avg_grad_norm = self._avg_grad_norm()
            scale = group["rho"] / (avg_grad_norm + 1e-7)

            for p in group["params"]:
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

                self.state[p]['g_{t-1}'] = self.state[p]['g_t'].clone()
                self.state[p]['g_t'] = p.grad.clone().detach()
        
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

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

        return output, loss
    

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