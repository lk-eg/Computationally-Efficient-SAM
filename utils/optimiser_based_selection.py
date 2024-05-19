# check if we need a closure
# for these, the reported TRAINING LOSS is the PERTURBED LOSS. Be AWARE of that.
def need_closure(args):
    return (
        args.opt[:3] == 'sam' 
        or args.opt[:5] == 'vasso' 
        or args.opt[:7] == 'vassore'
        or args.opt[:9] == 'vassoremu'
        or args.opt[:8] == 'adavasso'
    )

def optimiser_overhead_calculation(args):
    return (
        args.opt[:7] == 'vassore'
        or args.opt[:9] == 'vassoremu'
    )