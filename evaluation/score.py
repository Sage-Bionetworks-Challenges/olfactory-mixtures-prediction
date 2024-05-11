#!/usr/bin/env python3
"""Score prediction file.

Task 1 and 2 will return the same metrics:
    - RMSE
    - Pearson (two-sided, resampling method)
"""

import argparse
import json

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-g", "--goldstandard_file", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results.json")
    return parser.parse_args()


def score(gold, gold_col, pred, pred_col):
    """
    Calculate metrics for: RMSE, Pearson
    """
    rmse = root_mean_squared_error(gold[gold_col], pred[pred_col])
    pearson = pearsonr(gold[gold_col], pred[pred_col])

    return {
        "RMSE": rmse,
        "Pearson_correlation": pearson.statistic,
    }


def main():
    """Main function."""
    args = get_args()

    pred = pd.read_csv(args.predictions_file)
    gold = pd.read_csv(args.goldstandard_file)

    # Replace spaces in column headers in case they're found.
    gold.columns = [colname.replace(" ", "_") for colname in gold.columns]

    scores = score(gold, "Experimental_Values", 
                   pred, "Predicted_Experimental_Values")

    with open(args.output, "w") as out:
        res = {"submission_status": "SCORED", **scores}
        out.write(json.dumps(res))


if __name__ == "__main__":
    main()
