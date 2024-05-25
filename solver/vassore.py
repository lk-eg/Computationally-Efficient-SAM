import torch
import torch.optim
import random

from utils.configurable import configurable
from solver.build import OPTIMIZER_REGISTRY

from solver.vasso import VASSO
from solver.criteria_functions import criteria_functions


@OPTIMIZER_REGISTRY.register()
class VASSORE(VASSO):
    @configurable()
    def __init__(
        self,
        params,
        base_optimizer,
        logger,
        rho,
        theta,
        max_epochs,
        extensive_metrics_mode,
        performance_scores_mode,
        crt,
        crt_k,
        crt_p,
    ) -> None:
        super().__init__(
            params,
            base_optimizer,
            logger,
            rho,
            theta,
            max_epochs,
            extensive_metrics_mode,
            performance_scores_mode,
        )

        assert 0 <= crt_k and isinstance(crt_k, int), "k must be a natural number"
        assert 0 <= crt_p and crt_p <= 1, "p must live in [0, 1]."

        self.crt = crt
        self.k = crt_k
        self.p = crt_p
        self.rndm = 0.0
        self.inner_gradient_calculation_counter = 0

        if self.crt[:8] == "gSAMNorm":
            self.tau = 0
            for group in self.param_groups:
                for p in group["params"]:
                    itr_metric_keys = ["g_t"]
                    for key in itr_metric_keys:
                        self.state[p][key] = torch.zeros_like(
                            p, requires_grad=False
                        ).to(p)

        self.inner_gradient_calculation = criteria_functions[crt]
        self.g_norm = 0

    @classmethod
    def from_config(cls, args):
        config = super().from_config(args)
        config["crt"] = args.crt
        config["crt_k"] = args.crt_k
        config["crt_p"] = args.crt_p
        return config

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if self.inner_gradient_calculation(self):
            self._ema_update()

        avg_grad_norm = self._avg_grad_norm("ema")
        for group in self.param_groups:
            if self.inner_gradient_calculation(self):
                scale = group["rho"] / (avg_grad_norm + 1e-16)

            for p in group["params"]:
                if self.inner_gradient_calculation(self):
                    e_w = self._new_e_w_calculation(p, scale)
                else:
                    e_w = self.state[p]["e_t"]

                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_t"])

                if self.extensive_metrics_mode:
                    # I am running here an analysis on the outer gradient, g_{SAM}, not the inner gradient.
                    self.state[p]["g_{t-1}"] = self.state[p]["g_t"].clone()
                self.state[p]["g_t"] = p.grad

        if self.crt[:8] == "gSAMNorm":
            self.g_norm = self._avg_grad_norm("g_t").item()
            self.tau = (1 - self.theta) * self.tau + self.theta * self.g_norm

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        if self.crt == "random":
            self.rndm = random.random()

        computeForward = (
            self.performance_scores_mode or self.inner_gradient_calculation(self)
        )
        computeBackprop = self.inner_gradient_calculation(self)
        self.inner_gradient_calculation_counter += computeBackprop
        with torch.enable_grad():
            if computeForward:
                innerOutput, innerLoss = closure(computeForward, computeBackprop)
            else:
                closure(computeForward, computeBackprop)
        self.first_step()
        with torch.enable_grad():
            outerOutput, outerLoss = closure(True, True)
        self.second_step()

        self.iteration_step_counter += 1

        # With full knowledge that this is worse than innerOutput, innerLoss.
        # Reasoning: see closure() definition in utils/engine.py
        return outerOutput, outerLoss
