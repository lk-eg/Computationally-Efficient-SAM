# Check if the running optimizer needs a closure
# For such optimizers, the reported TRAINING LOSS is the PERTURBED LOSS
def need_closure_fn(args):
    return (
        args.opt[:3] == "sam"
        or args.opt[:5] == "vasso"
        or args.opt[:7] == "vassore"
        or args.opt[:9] == "vassoremu"
        # or args.opt[:8] == "adavasso"
    )


def hyperparameters(args):
    hyperparameter_string = f"lr={args.lr}"
    if need_closure_fn(args):
        hyperparameter_string += f"rho={args.rho}"
    if args.opt[:5] == "vasso":
        hyperparameter_string += f"theta={args.theta}"

    # more cases have to be added later
    return hyperparameter_string


# Re: Scheduling optimizer
# find the epochs in which VaSSO will be scheduled
def schedule_epoch_ranges(input_string):
    elements = input_string.split(",")
    epoch_ranges = []
    for element in elements:
        if "-" in element:
            start, end = map(int, element[1:-1].split("-"))
            epoch_ranges.append((start, end))
        else:
            epoch = int(element)
            epoch_ranges.append((epoch, epoch))
    return epoch_ranges


# return true if the current epoch runs VaSSO
def scheduling(current_epoch, epoch_ranges):
    for start, end in epoch_ranges:
        if start <= current_epoch < end:
            return True
    return False


# Check whether we should calculate overhead over baseline SGD
def optimiser_overhead_calculation(args):
    return args.crt != "baseline"
