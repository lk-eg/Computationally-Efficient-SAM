import pandas as pd
from solver.criteria_functions import criteria_parameters
from utils.optimiser_based_selection import (
    optimiser_overhead_calculation,
    hyperparameters,
)
import csv
import fcntl


# Write global comparison txt file
def comp_file_logging(
    args,
    max_acc,
    train_acc1,
    test_loss,
    train_loss,
    overhead,
    used_training,
    optimiser_overhead_calculation,
):
    comp_file_name = "comp_" + args.dataset_nn_combination + ".info"
    with open(comp_file_name, "a") as f:
        opt = args.opt.split("-")[0]
        f.write(f"OPTIMISER: {opt} - theta = {args.theta} \n")
        if optimiser_overhead_calculation:
            reuse_method = args.crt
            f.write(f"Reuse method: {reuse_method}")
            if reuse_method == "naive":
                f.write(f" - k={args.crt_k} \n")
            elif reuse_method == "random":
                f.write(f" - p={args.crt_p} \n")
            elif reuse_method[:11] == "gSAMNormEMA":
                f.write(f" - crt_z={args.crt_z} \n")
        f.write(
            f"Max Test Accuracy: {max_acc:.4f}, Last Train Accuracy: {train_acc1:.4f}, Difference (Train Accuracy - Test Accuracy) = {train_acc1 - max_acc:.4f} \n"
        )
        f.write(
            f"Last Test Loss: {test_loss:.4f}, Last Train Loss: {train_loss:.4f}, Difference (test loss - train loss) = {test_loss - train_loss:.4f} \n"
        )
        if optimiser_overhead_calculation:
            f.write(f"Backprop Overhead x SGD: {overhead:.2f} \n")
        f.write(f"Training Time: {used_training} \n")
        f.write("\n")


# Results Collection
# Save results data of every experiment into a csv file
def training_result_save(
    args,
    top_1_test_acc,
    overfitting_indicator,
    fwp_overhead_over_sgd,
    bwp_overhead_over_sgd,
    images_per_sec,
    runtime,
    max_allocated_memory,
    max_reserved_memory,
    lambda_1=None,
    lambda_5=None,
):
    criterion = args.crt
    results = []
    if lambda_1 is not None:
        jastr = round(lambda_1 / lambda_5, 4)
    else:
        jastr = None

    if optimiser_overhead_calculation(args):
        criterium_parameter = criteria_parameters(args, criterion)
    else:
        criterium_parameter = None

    exp_res = {
        "optimizer": args.opt,
        "hyperparameters": hyperparameters(args),
        "criterion": criterion,
        "crt_parameter": criterium_parameter,
        "top-1 test acc": top_1_test_acc,
        "overfitting indicator": round(overfitting_indicator, 4),
        "l1": lambda_1,
        "l5": lambda_5,
        "l1/l5": jastr,
        "fwp_overhead": round(fwp_overhead_over_sgd, 4),
        "bwp_overhead": round(bwp_overhead_over_sgd, 4),
        "images/s": round(images_per_sec, 2),
        "runtime": round(runtime, 2),
        "max_allocated_memory": max_allocated_memory,
        "max_reserved_memory": max_reserved_memory,
        "epochs": args.epochs,
        "exclusive_run": args.exclusive_run,
    }
    if args.crt == "none":
        exp_res["criterion"] = "none"
        exp_res["crt_parameter"] = "none"
    results.append(exp_res)
    df = pd.DataFrame(results)
    numerical_results_csv_fp = args.dataset_nn_combination + "_results.csv"
    try:
        with open(numerical_results_csv_fp, "r"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    df.to_csv(
        numerical_results_csv_fp,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )


# As I want to know the distribution of gradient norms
def decision_rule_save(args, optimizer):
    if args.crt[:4] == "gSAM":
        gSAMema = optimizer.gSAMema
        gSAMnorm_values = [entry["gSAMnorm"] for entry in gSAMema]
        tau_values = [entry["tau"] for entry in gSAMema]
        with open("gSAMstudy.csv", "a", newline="") as file:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                writer = csv.writer(file)
                writer.writerow(gSAMnorm_values)
                writer.writerow(tau_values)
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    if args.crt == "cosSim":
        cossim_values = optimizer.cosSims
        with open("cosSimsstudy.csv", "a", newline="") as file:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                writer = csv.writer(file)
                writer.writerow(cossim_values)
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    criterion_logger = optimizer.criterion_logger
    if args.crt[:4] == "gSAM":
        name = f"crt={args.crt}_lam={args.lam}_z1={args.crt_z}_z2={args.z_two}"
    elif args.crt == "cosSim":
        name = f"crt=cosSim_c={args.crt_c}_seed={args.seed}"
    criterion_logger.insert(0, name)
    file_name = f"criterion_logger_{args.crt}.csv"
    with open(file_name, "a", newline="") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            writer = csv.writer(file)
            writer.writerow(criterion_logger)
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
