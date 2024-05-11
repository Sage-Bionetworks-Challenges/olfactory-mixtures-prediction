#!/usr/bin/env python3
"""Validate prediction file.

Prediction file should be a 4-col CSV file.
"""

import argparse
import json

import pandas as pd

INDEX = ["Dataset", "Mixture_1", "Mixture_2"]
COLS = {
    "Dataset": str,
    "Mixture_1": int,
    "Mixture_2": int,
    "Predicted_Experimental_Values": float,
}


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-g", "--goldstandard_file", type=str, required=True)
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


def check_dups(pred):
    """Check for duplicate mixtures."""
    duplicates = pred.index.duplicated()
    if duplicates.any():
        return (
            f"Found {duplicates.sum()} duplicate mixture(s): "
            f"{pred[duplicates].index.to_list()}"
        )
    return ""


def check_missing_mixtures(gold, pred):
    """Check for missing mixtures."""
    missing_ids = gold.index.difference(pred.index)
    if len(missing_ids) > 0:
        return (
            f"Found {missing_ids.shape[0]} missing mixture(s): "
            f"{missing_ids.to_list()}"
        )
    return ""


def check_unknown_mixtures(gold, pred):
    """Check for unknown mixtures."""
    unknown_ids = pred.index.difference(gold.index)
    if len(unknown_ids) > 0:
        return (
            f"Found {unknown_ids.shape[0]} unknown mixture(s): "
            f"{unknown_ids.to_list()}"
        )
    return ""


def check_nan_values(pred, col):
    """Check for NAN predictions."""
    missing_probs = pred[col].isna().sum()
    if missing_probs:
        return f"'{col}' column contains {missing_probs} NaN value(s)."
    return ""


def check_prob_values(pred, col):
    """Check that probabilities are between [0, 1]."""
    if (pred[col] < 0).any() or (pred[col] > 1).any():
        return f"'{col}' column should be between [0, 1] inclusive."
    return ""


def validate(gold_file, pred_file):
    """Validate predictions file against goldstandard."""
    errors = []

    gold = pd.read_csv(gold_file)

    # Replace spaces in column headers in case they're found.
    gold.columns = [colname.replace(" ", "_") for colname in gold.columns]
    gold.set_index(INDEX, inplace=True)

    try:
        pred = pd.read_csv(
            pred_file,
            usecols=COLS,
            dtype=COLS,
            float_precision="round_trip",
            index_col=INDEX,
        )
    except ValueError:
        errors.append(
            "Invalid prediction file headers and/or column types. "
            f"Expecting: {str(COLS)}."
        )
    else:
        errors.append(check_dups(pred))
        errors.append(check_missing_mixtures(gold, pred))
        errors.append(check_unknown_mixtures(gold, pred))
        errors.append(check_nan_values(pred, "Predicted_Experimental_Values"))
        errors.append(check_prob_values(pred, "Predicted_Experimental_Values"))
    return errors


def main():
    """Main function."""
    args = get_args()

    invalid_reasons = validate(
        gold_file=args.goldstandard_file, pred_file=args.predictions_file
    )

    invalid_reasons = "\n".join(filter(None, invalid_reasons))
    status = "INVALID" if invalid_reasons else "VALIDATED"

    # truncate validation errors if >500 (character limit for sending email)
    if len(invalid_reasons) > 500:
        invalid_reasons = invalid_reasons[:496] + "..."
    res = json.dumps(
        {"submission_status": status, "submission_errors": invalid_reasons}
    )

    if args.output:
        with open(args.output, "w") as out:
            out.write(res)
    else:
        print(res)


if __name__ == "__main__":
    main()
