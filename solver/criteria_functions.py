WARMUP_CONSTANT = 100

"""
CRITERIA that tell us if inner gradient has to be calculated
"""
def naive_criterion(self):
    return self.iteration_step_counter % self.k == 0 or self.iteration_step_counter <= WARMUP_CONSTANT
    
def random_criterion(self):
    return self.rndm < self.p or self.iteration_step_counter <= WARMUP_CONSTANT
    
def gSAMNormEMA_criterion(self):
    return self.tau <= self.g_norm or self.iteration_step_counter <= WARMUP_CONSTANT
    
def gSAMNormEMAInverted_criterion(self):
    return not gSAMNormEMA_criterion(self) or self.iteration_step_counter <= WARMUP_CONSTANT

# collect all possible criteria for inner gradient calculation
criteria_functions = {
    'naive': naive_criterion,
    'random': random_criterion,
    'gSAMNormEMA': gSAMNormEMA_criterion,
    'gSAMNormEMAInverted': gSAMNormEMAInverted_criterion,
}