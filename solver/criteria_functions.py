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


def gSAMNormEMA_criterion(self):
    return (
        self.tau <= self.zeta * self.g_norm
        or self.iteration_step_counter <= WARMUP_CONSTANT
    )


def gSAMNormEMAInverted_criterion(self):
    return (
        not gSAMNormEMA_criterion(self)
        or self.iteration_step_counter <= WARMUP_CONSTANT
    )


def scheduling_block(self, calculation_range_set):
    return self.iteration_step_counter in calculation_range_set


# collect all possible criteria for inner gradient calculation
criteria_functions = {
    "naive": naive_criterion,
    "random": random_criterion,
    "gSAMNormEMA": gSAMNormEMA_criterion,
    "gSAMNormEMAInverted": gSAMNormEMAInverted_criterion,
}

criteria_parameter_names = {
    "naive": ("k", "crt_k"),
    "random": ("p", "crt_p"),
    "schedule_endblock": ("b", "crt_b"),
    "gSAMNormEMA": ("z", "zeta"),
    "gSAMNormEMAInveted": ("z", "zeta"),
}


def criteria_parameters(args, criterion):
    crt_keyword = criteria_parameter_names[criterion][1]
    crt_value = getattr(args, crt_keyword)
    return f"{criteria_parameter_names[criterion][0]}={crt_value}"
