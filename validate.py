#!/usr/bin/env python3
"""Validate prediction file.

Prediction file should be a 54-col CSV file.
"""

import argparse
import json
import os
import re
from glob import glob

import pandas as pd

INDEX_COL = "stimulus"


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-g", "--goldstandard_folder", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results.json")
    parser.add_argument("-t", "--task", type=int, default=1)
    return parser.parse_args()


def extract_gs_file(folder):
    """Extract goldstandard file from folder."""
    files = glob(os.path.join(folder, "*"))

    # Filter out the manifest file
    files = [f for f in files if os.path.basename(f) != "SYNAPSE_METADATA_MANIFEST.tsv"]

    if len(files) != 1:
        raise ValueError(
            "Expected exactly one goldstandard file in folder. "
            f"Got {len(files)}. Exiting."
        )
    return files[0]


def check_dups(pred):
    """Check for duplicate IDs."""
    duplicates = pred.index.duplicated()
    if duplicates.any():
        return (
            f"Found {duplicates.sum()} duplicate stimulus ID(s): "
            f"{pred[duplicates].index.to_list()}"
        )
    return ""


def check_missing_ids(gold, pred):
    """Check for missing IDs."""
    missing_ids = gold.index.difference(pred.index)
    if len(missing_ids) > 0:
        return (
            f"Found {missing_ids.shape[0]} missing stimulus ID(s): "
            f"{missing_ids.to_list()}"
        )
    return ""


def check_unknown_ids(gold, pred):
    """Check for unknown IDs."""
    unknown_ids = pred.index.difference(gold.index)
    if len(unknown_ids) > 0:
        return (
            f"Found {unknown_ids.shape[0]} unknown stimulus ID(s): "
            f"{unknown_ids.to_list()}"
        )
    return ""


def check_nan_values(pred):
    """Check for NAN predictions."""
    # Check if all values are NaN.
    if pred.isna().all().all():
        return "All columns contain NaN values."
    return ""


def check_prob_values(pred):
    """Check that probabilities are between [0, 5]."""
    issue_message = []
    for col in pred.columns[1:]:
        if (pred[col] < 0).any() or (pred[col] > 5).any():
            issue_message.append(f"'{col}' column should be between [0, 5] inclusive.")
    return "\n".join(issue_message) if issue_message else ""


def validate(gold_file, pred_file):
    """Validate predictions file against goldstandard."""
    errors = []

    gold = pd.read_csv(gold_file).set_index(INDEX_COL)
    expected_cols = (
        gold.reset_index().drop(["Intensity", "Pleasantness"], axis=1).dtypes.to_dict()
    )

    try:
        pred = pd.read_csv(
            pred_file,
            usecols=expected_cols,
            dtype=expected_cols,
            float_precision="round_trip",
        ).set_index(INDEX_COL)
    except ValueError:
        errors.append(
            "Invalid prediction file headers and/or column types. "
            "Expecting `stimulus` (str) and 51 semantic descriptors (float). "
            "See the Task 1 - Data page for more details."
        )
    else:
        errors.append(check_dups(pred))
        errors.append(check_missing_ids(gold, pred))
        errors.append(check_unknown_ids(gold, pred))
        errors.append(check_nan_values(pred))
        errors.append(check_prob_values(pred))
    return errors


def main():
    """Main function."""
    args = get_args()

    invalid_reasons = validate(
        gold_file=extract_gs_file(args.goldstandard_folder), pred_file=args.predictions_file
        )

    invalid_reasons = "\n".join(filter(None, invalid_reasons))
    status = "INVALID" if invalid_reasons else "VALIDATED"
    # Identify words that will require variations in output truncation
    trigger_words = ["missing", "unknown"]
    pattern = r"\b(" + "|".join(map(re.escape, trigger_words)) + r")\b"

    # Split the string into individual lines (reasons)
    lines = invalid_reasons.splitlines()

    # truncate validation errors if >500 (character limit for sending email)
    if len(invalid_reasons) > 500:
        if any(re.search(pattern, line) for line in lines):
            # If any line contains trigger words, truncate to the first 3 lines
            invalid_reasons = "\n".join(lines[:3]) + "..."
        elif any(not re.search(pattern, line) for line in lines):
            # If any line does not contain trigger words, truncate to 496 characters
            invalid_reasons = invalid_reasons[:496] + "..."
    # Clean up float-heavy tuples (if present in stringified form)
    invalid_reasons = re.sub(r"\(\s*'([^']+)'(?:,.*?)*\)", r"'\1'", invalid_reasons)
    res = json.dumps(
        {"validation_status": status, 
        "validation_errors": re.sub(r"\(\s*'([^']+)'(?:,.*?)*\)", r"'\1'", invalid_reasons)}
    )

    # print the results to a JSON file
    with open(args.output, "w") as out:
        out.write(res)
    # print the validation status to STDOUT
    print(status)


if __name__ == "__main__":
    main()
