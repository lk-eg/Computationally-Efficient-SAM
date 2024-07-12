# check if we need a closure
# for these, the reported TRAINING LOSS is the PERTURBED LOSS. Be AWARE of that.
def need_closure_fn(args):
    return (
        args.opt[:3] == "sam"
        or args.opt[:5] == "vasso"
        or args.opt[:7] == "vassore"
        or args.opt[:9] == "vassoremu"
        # or args.opt[:8] == "adavasso"
    )


def scheduling(input_string, current_epoch):
    elements = input_string.split(",")
    epoch_ranges = []
    for element in elements:
        if "-" in element:
            start, end = map(int, element.split("-"))
            epoch_ranges.append((start - 1, end - 1))
        else:
            epoch = int(element)
            epoch_ranges.append((epoch - 1, epoch))
    for start, end in epoch_ranges:
        if start <= current_epoch < end:
            return True
    return False


def optimiser_overhead_calculation(args):
    return args.crt != "baseline"
