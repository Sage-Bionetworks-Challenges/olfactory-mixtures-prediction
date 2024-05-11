"""Python Model Example"""

import os

import typer
import pandas as pd
import numpy as np


def predict(df):
    """
    Run a "prediction": generate random floats between [0.0, 1.0)
    """
    nrows = len(df.index)
    df["Predicted_Experimental_Values"] = np.random.random_sample(size=nrows)
    return df


def main(input_dir: str = "/input", output_dir: str = "/output"):
    """
    Accept two params: `input_dir`, `output_dir`
    """
    # data = pd.read_csv(os.path.join(input_dir, "names.csv"))
    data = pd.read_csv("Leaderboard_set_Submission_form.csv")
    predictions = predict(data)
    predictions.to_csv(os.path.join(output_dir, "predictions.csv"),
                       index=False)


if __name__ == "__main__":
    typer.run(main)
