import pandas as pd
import numpy as np
from scipy import stats

EXPERIMENTS_CSV = "cifar100_wrn-28-10"
file_name = EXPERIMENTS_CSV + "_results.csv"


# Create Latex Tables to compare optimizers
def latex_table_creation():
    df = pd.read_csv(file_name)
    grouped = df.groupby(["optimizer", "criterion"])
    summary = grouped["top-1 test acc"].agg(["mean", "count", "std"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["conf_interval"] = summary["sem"] * stats.t.ppf(
        (1 + 0.95) / 2, summary["count"] - 1
    )
