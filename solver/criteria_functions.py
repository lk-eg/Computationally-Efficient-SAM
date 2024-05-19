WARMUP_CONSTANT = 100

"""
CRITERIA that tell us if inner gradient has to be calculated
"""
def naive_criterion(self):
    return self.iteration_step_counter <= WARMUP_CONSTANT or self.iteration_step_counter % self.k == 0
    
def random_criterion(self):
    import random
    return self.iteration_step_counter <= WARMUP_CONSTANT and random.random() < self.p
    
def gSAMNormEMA_criterion(self):
    return self.iteration_step_counter <= WARMUP_CONSTANT and self.tau <= self.g_norm
    
def gSAMNormEMAInverted_criterion(self):
    return self.iteration_step_counter <= WARMUP_CONSTANT and not self.gSAMNormEMA_criterion()

# collect all possible criteria for inner gradient calculation
criteria_functions = {
    'naive': naive_criterion,
    'random': random_criterion,
    'gSAMNormEMA': gSAMNormEMA_criterion,
    'gSAMNormEMAInverted': gSAMNormEMAInverted_criterion,
}