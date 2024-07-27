import pandas as pd
import numpy as np
from scipy import stats
from operator import itemgetter

EXPERIMENTS_CSV = "cifar100_wrn28-10_mass"
file_name = EXPERIMENTS_CSV + "_results.csv"

data = pd.read_csv(file_name)

baseline_data = data[data["criterion"] == "baseline"]
random_data = data[data["criterion"] == "random"]
naive_data = data[data["criterion"] == "naive"]
schedule_data = data[data["criterion"] == "schedule"]


def mean_ci(series):
    mean = series.mean()
    ci = stats.sem(series) * stats.t.ppf((1 + 0.95) / 2.0, len(series) - 1)
    return mean, ci


# grouped = baseline_data.groupby(["optimizer", "hyperparameters"])
grouped = schedule_data.groupby(["optimizer", "crt_parameter"])


results = []
for (optimizer, crt_parameter), group in grouped:
    top1_acc_mean, top1_acc_ci = mean_ci(group["top-1 test acc"])
    overfitting_mean, overfitting_ci = mean_ci(group["overfitting indicator"])
    bwp_overhead = group["bwp_overhead"].mean()

    exclusive_run_group = group[group["exclusive_run"]]
    if not exclusive_run_group.empty:
        images_per_s = exclusive_run_group["images/s"].mean()
        runtime = exclusive_run_group["runtime"].mean()
        max_allocated_memory = exclusive_run_group["max_allocated_memory"].mean()
    else:
        images_per_s = runtime = max_allocated_memory = np.nan

    results.append(
        {
            "optimizer": optimizer,
            "crt_parameter": crt_parameter,
            "top1_acc_mean": top1_acc_mean,
            "top1_acc_ci": top1_acc_ci,
            "overfitting_mean": overfitting_mean,
            "overfitting_ci": overfitting_ci,
            "bwp_overhead": bwp_overhead,
            "images_per_s": images_per_s,
            "runtime": runtime,
            "max_allocated_memory": max_allocated_memory,
        }
    )

# results = sorted(results, key=itemgetter("optimizer", "crt_parameter"))
# results = sorted(results, key=itemgetter("crt_parameter"), reverse=True)
results_df = pd.DataFrame(results)


def create_latex_table(df):
    latex_str = """
        \\begin{table}[h!]
        \\centering
        \\resizebox{\\textwidth}{!}{%
        \\begin{tabular}{|| l || c | c | c | c | c | c |} 
        \\hline 
        Optimizer & s= & Test Acc & Overfit & BWP $\\times$ SGD & Runtime [min] & Max Mem [MB]  \\\\
        \\hline 
        \\hline"""

    for _, row in df.iterrows():
        latex_str += f"{row['optimizer']} & {row['crt_parameter']} & {row['top1_acc_mean']:.2f}$_{{\\pm {row['top1_acc_ci']:.2f}}}$ & "
        latex_str += f"{row['overfitting_mean']:.2f}$_{{\\pm {row['overfitting_ci']:.2f}}}$ & {row['bwp_overhead']:.2f} & "
        latex_str += f"{row['runtime']:.2f} & {row['max_allocated_memory']:.2f} \\\\\n"

    latex_str += """\\hline
        \\end{tabular}}
        \\caption{Experiment Results for naive}
        \\label{tab:naive_approach_results}
        \\end{table}"""

    return latex_str


latex_table = create_latex_table(results_df)
print(latex_table)
