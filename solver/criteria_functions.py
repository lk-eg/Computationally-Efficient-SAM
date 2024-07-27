import numpy as np

WARMUP_CONSTANT = 1
WARMUP_CONSTANT_MEAN_BASED_METHODS = 100

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
    criterion_trigger = self.tau < self.phi_prime * self.g_norm
    self.criterion_logger.append(criterion_trigger + 0)
    return (
        criterion_trigger
        or self.iteration_step_counter <= WARMUP_CONSTANT_MEAN_BASED_METHODS
    )


def gSAMflat_criterion(self):
    criterion_trigger = self.tau > self.phi * self.g_norm
    self.criterion_logger.append(criterion_trigger + 0)
    return (
        criterion_trigger
        or self.iteration_step_counter <= WARMUP_CONSTANT_MEAN_BASED_METHODS
    )


def gSAMratio_criterion(self):
    criterion_trigger = (
        self.tau > self.phi * self.g_norm or self.tau < self.phi_prime * self.g_norm
    )
    self.criterion_logger.append(criterion_trigger + 0)
    return (
        criterion_trigger
        or self.iteration_step_counter <= WARMUP_CONSTANT_MEAN_BASED_METHODS
    )


def cosSim_criterion(self):
    cos_sim = self._cosine_similarity("g_t", "g_{t-1}")
    if not self.crt_c:
        criterion_trigger = cos_sim < 0
    else:
        criterion_trigger = cos_sim > 0
    self.cosSims.append(cos_sim)
    self.criterion_logger.append(criterion_trigger + 0)
    return criterion_trigger or self.iteration_step_counter <= WARMUP_CONSTANT


def variance_criterion(self):
    if self.iteration_step_counter <= WARMUP_CONSTANT:
        reset_stat_metrics(self)
        return True
    elif not self.iteration_step_counter % 100:
        reset_stat_metrics(self)
        return True
    elif self.var_gsam_norm >= self.var_delta:
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
    "gSAMratio": gSAMratio_criterion,
    "variance": variance_criterion,
    "cosSim": cosSim_criterion,
    "chebyshev": chebyshev_criterion,
}

criteria_parameter_names = {
    "naive": ("k", "crt_k"),
    "random": ("p", "crt_p"),
    "gSAMsharp": ("z", "crt_z"),
    "gSAMflat": ("z", "crt_z"),
    "gSAMratio": ("z", "crt_z"),
    "schedule": ("s", "crt_s"),
    "variance": ("v", "var_delta"),
    "cosSim": (">0?", "crt_c"),
}


def criteria_parameters(args, criterion):
    crt_keyword = criteria_parameter_names[criterion][1]
    crt_value = getattr(args, crt_keyword)
    return f"{criteria_parameter_names[criterion][0]}={crt_value}"
