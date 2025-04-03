#!/usr/bin/env python3
"""Score prediction file.

Task 1 and 2 will return the same metrics:
    - RMSE
    - Pearson (two-sided, resampling method)
"""

import argparse
from glob import glob
import json
import os
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
    pearson = pearsonr(gold[gold_col], pred[pred_col]).statistic

    return {
        "RMSE": rmse,
        "Pearson_correlation": None if pd.isna(pearson) else pearson
    }


def extract_gs_file(folder):
    """Extract goldstandard file from folder."""
    files = glob(os.path.join(folder, "*"))

    # Filter out the manifest file
    files = [f for f in files if os.path.basename(f) != 'SYNAPSE_METADATA_MANIFEST.tsv']

    if len(files) != 1:
        raise ValueError(
            "Expected exactly one goldstandard file in folder. "
            f"Got {len(files)}. Exiting."
        )
    return files[0]


def check_validation_status(filename, args):
    """
    Determine whether the validate.py script successfully validated the data.
    """
    
    with open(filename, "r") as f:
        status_result = json.load(f)
    print("Checking the validation_status.")
    
    if status_result.get("validation_status") == "INVALID":
        new_data = {"validation_status": "INVALID",
                    "validation_errors": status_result.get("validation_errors"),
                    "score_status":  "INVALID",
                    "score_errors": "Submission could not be evaluated due to validation errors."}
    
        with open(args.output, "w") as out:
            out.write(json.dumps(new_data))

        print(status_result.get("validation_status"))
    else:
        print("Scoring the submission.")
        
        pred = pd.read_csv(args.predictions_file)
        gold = pd.read_csv(extract_gs_file(args.goldstandard_file))

        # Replace spaces in column headers in case they're found.
        gold.columns = [colname.replace(" ", "_") for colname in gold.columns]

        # Strip leading and trailing spaces as needed.
        gold = gold.map(lambda x: x.strip() if isinstance(x, str) else x)
        pred = pred.map(lambda x: x.strip() if isinstance(x, str) else x)

        scores = score(gold, "Experimental_Values",
                        pred, "Predicted_Experimental_Values")
        
        with open(args.output, "w") as out:
            res = {"validation_status": status_result.get("validation_status"),
                    "validation_errors": status_result.get("validation_errors"),
                    "score_status": "SCORED", "score_errors": "", **scores,
                    }
            out.write(json.dumps(res))
        print("SCORED")


def main():
    """Main function."""
    args = get_args()

    check_validation_status(args.output, args)

if __name__ == "__main__":
    main()
