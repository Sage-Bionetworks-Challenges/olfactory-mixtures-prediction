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
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-g", "--goldstandard_folder", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results.json")
    parser.add_argument("-t", "--task", type=int, default=1)
    return parser.parse_args()


def evaluate_submission(pred, gold):
    """Rank and calculate average Pearson correlation and cosine distance."""
    pred_df = pd.read_csv(pred).sort_values("stimulus").reset_index(drop=True)
    gold_df = pd.read_csv(gold).sort_values("stimulus").reset_index(drop=True)
    feature_cols = pred_df.columns.difference(["stimulus"])

    pearson_scores = []
    cosine_dists = []

    for i in range(len(pred_df)):
        pred_vector = pred_df.loc[i, feature_cols].values
        true_vector = gold_df.loc[i, feature_cols].values

        pearson_corr, _ = pearsonr(pred_vector.astype(float), true_vector.astype(float))
        cosine_dist = cosine_distances([pred_vector], [true_vector])[0, 0]

        pearson_scores.append(pearson_corr)
        cosine_dists.append(cosine_dist)

    return {
        "pearson_correlation": np.mean(pearson_scores),
        "cosine": np.mean(cosine_dists)
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
    Stop scoring if previous validations fail or if there is a scoring error.
    """
    with open(filename, "r") as f:
        status_result = json.load(f)

    # Initialize the result dictionary
    res = {
        "validation_status": status_result.get("validation_status"),
        "validation_errors": status_result.get("validation_errors"),
    }

    if status_result.get("validation_status") == "INVALID":
        status = status_result.get("validation_status")
        errors = "Validation failed. Submission not scored."
        scores = {}
    else:
        try:
            scores = evaluate_submission(
            args.predictions_file, extract_gs_file(args.goldstandard_folder)
            )
            status = "SCORED"
            errors = ""
        except ValueError:
            status = "INVALID"
            errors = "Error encountered during scoring; submission not evaluated."
            scores = {}
        # To be made available once the scoring metrics decided for both tasks
        # except KeyError:
        #     errors = f"Invalid challenge task number specified: `{task_number}`"

    # Merge the existing result dictionary with additional outputs
    res |= {"score_status": status,
            "score_errors": errors,
            **scores,
        }
    
    with open(args.output, "w", encoding="utf-8") as out:
        out.write(json.dumps(res))

    print(status)


def main():
    """Main function."""
    args = get_args()

    check_validation_status(args.output, args)

if __name__ == "__main__":
    main()
