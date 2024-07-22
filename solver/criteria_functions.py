import numpy as np

WARMUP_CONSTANT = 100

"""
CRITERIA that tell us if inner gradient has to be calculated
"""


def naive_criterion(self):
    return (
        self.iteration_step_counter % self.k == 0
        or self.iteration_step_counter <= WARMUP_CONSTANT
    )


def random_criterion(self):
    return self.rndm < self.p or self.iteration_step_counter <= WARMUP_CONSTANT


def gSAMsharp_criterion(self):
    return (
        self.tau <= self.crt_z * self.g_norm
        or self.iteration_step_counter <= WARMUP_CONSTANT
    )


def gSAMflat_criterion(self):
    return (
        not gSAMsharp_criterion(self) or self.iteration_step_counter <= WARMUP_CONSTANT
    )


def scheduling_block(self, calculation_range_set):
    return self.iteration_step_counter in calculation_range_set


def variance_criterion(self):
    if self.iteration_step_counter <= WARMUP_CONSTANT:
        reset_stat_metrics(self)
        self.logger.wandb_log_batch(
            **{
                "decision_type": 0,
                "global_batch_counter": self.iteration_step_counter,
            }
        )
        return True
    elif not self.iteration_step_counter % 100:
        reset_stat_metrics(self)
        self.logger.wandb_log_batch(
            **{
                "decision_type": 1,
                "global_batch_counter": self.iteration_step_counter,
            }
        )
        return True
    elif self.var_gsam_norm >= self.var_delta:
        # 3 is a guessed threshold value. I could make it a hyperparameter. What would be a good threshold value?
        # It should rather be a hyperparameter
        self.logger.wandb_log_batch(
            **{
                "decision_type": 2,
                "global_batch_counter": self.iteration_step_counter,
            }
        )
        return True
    return False


def chebyshev_criterion(self):
    if self.iteration_step_counter <= WARMUP_CONSTANT:
        reset_stat_metrics(self)
        self.logger.wandb_log_batch(
            **{
                "decision_type": 0,
                "global_batch_counter": self.iteration_step_counter,
            }
        )
        return True
    elif not self.iteration_step_conuter % 100:
        reset_stat_metrics(self)
        self.logger.wandb_log_batch(
            **{
                "decision_type": 1,
                "global_batch_counter": self.iteration_step_counter,
            }
        )
        return True
    # By Chebyshev, the probability for the following event is bounded by 0.5
    elif abs(self.g_norm - self.mean_gsam_norm) >= np.sqrt(2) * np.sqrt(
        self.var_gsam_norm
    ):
        self.logger.wandb_log_batch(
            **{
                "decision_type": 2,
                "global_batch_counter": self.iteration_step_counter,
            }
        )
        self.decision_rule_counter += 1
        return True
    return False


def reset_stat_metrics(self):
    self.sum_gsam_norm = 0
    self.sum_gsam_norm_squared = 0
    self.mean_gsam_norm = 0
    self.var_gsam_norm = 0


# collect all possible criteria for inner gradient calculation
criteria_functions = {
    "naive": naive_criterion,
    "random": random_criterion,
    "gSAMsharp": gSAMsharp_criterion,
    "gSAMflat": gSAMflat_criterion,
    "variance": variance_criterion,
    "chebyshev": chebyshev_criterion,
}

criteria_parameter_names = {
    "naive": ("k", "crt_k"),
    "random": ("p", "crt_p"),
    "schedule": ("s", "crt_s"),
    "gSAMsharp": ("z", "crt_z"),
    "gSAMflat": ("z", "crt_z"),
}


def criteria_parameters(args, criterion):
    crt_keyword = criteria_parameter_names[criterion][1]
    crt_value = getattr(args, crt_keyword)
    return f"{criteria_parameter_names[criterion][0]}={crt_value}"
