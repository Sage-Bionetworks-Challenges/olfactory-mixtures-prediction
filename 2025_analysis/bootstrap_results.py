"""Bootstrap Submission Scores by Task.

This script will bootstrap submissions by task made to the DREAM Olfactory Mixtures
Prediction Challenge 2025 - (syn64743570) according to
Pearson Correlation and Cosine. Only final round submissions are considered.
Only the latest submission for each submitter is kept.
A team was given perission to have a submission that is not the latest included.
"""
from datetime import datetime
from glob import glob
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
# Merge all team prediction columns together may be eliminated if no longer needed
# from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import synapseclient


SUBMISSION_VIEWS = {
    "Task 1": "syn66279193",
    "Task 2": "syn66484079"
}

INDEX_COL = "stimulus"

def get_args():
    """Set up command-line interface and get arguments."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--goldstandard_folder", type=str, required=True)
    parser.add_argument("-t", "--task", type=int, default=1)
    parser.add_argument("-o", "--output_prefix", type=str, default="olfactory-mixtures-prediction_BF")
    parser.add_argument("-n", "--n_bootstraps", type=int, default=10000)
    parser.add_argument("-s", "--sample_pct", type=float, default=0.1)
    parser.add_argument("-m", "--metric", type=str, choices=["rmse", "pearson"], default="rmse")
    parser.add_argument("--subview_id", type=str, default=None)
    # either "Final Round DREAM Olfactory Mixtures Prediction Challenge 2025 - Task 1" or "Final Round DREAM Olfactory Mixtures Prediction Challenge 2025 - Task 2"
    parser.add_argument("--evaluation_id", type=str, default=None)
    args = parser.parse_args()
    if args.evaluation_id is None:
        args.evaluation_id = (
            "Final Round DREAM Olfactory Mixtures Prediction Challenge 2025 - Task 1"
            if args.task == 1
            else "Final Round DREAM Olfactory Mixtures Prediction Challenge 2025 - Task 2"
        )
    return args
    
def extract_gs_file(folder: str) -> str:
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

def select_final_round_submissions(syn: synapseclient.Synapse, subview_id: str, evaluation_id: str) -> pd.DataFrame:
    """
    Get final round submissions from synapse tables that include all submissions for both rounds.
    Outputs a final averaged rank and sorted leaderboard.
    Only submissions from the specified evaluation are considered.
    For each submitter, but one team, only the submission closest to August 8, 2025 is kept.
    If any of the IDs [9756929, 9756930, 9756939, 9756938, 9756943, 9756942] are present,
    keep only 9756929 if present, otherwise keep only 9756930 if present, and remove the rest.
    """

    query = (
        f"SELECT id, pearson_correlation, cosine, createdOn, submitterid FROM {subview_id} "
        f"WHERE score_status = 'SCORED' "
        f"AND evaluationid = '{evaluation_id}'"
    ) # query_df should have columns: 'id' and 'submitterid'
    submissions = syn.tableQuery(query).asDataFrame()

    # Special handling for the specified IDs
    replace_ids = [9756929, 9756930, 9756939, 9756938, 9756943, 9756942]
    present_ids = [rid for rid in replace_ids if rid in submissions['id'].values]
    if present_ids:
        if 9756929 in present_ids:
            submissions = submissions[~submissions['id'].isin(replace_ids) | (submissions['id'] == 9756929)]
        elif 9756930 in present_ids:
            submissions = submissions[~submissions['id'].isin(replace_ids) | (submissions['id'] == 9756930)]
        else:
            submissions = submissions[~submissions['id'].isin(replace_ids)]

    # Find the submission closest to August 8, 2025 for each submitter
    target_date = int(pd.Timestamp("2025-08-08T23:59:59Z").timestamp() * 1000)
    submissions['createdOn_diff'] = np.abs(submissions['createdOn'] - target_date)
    submissions = submissions.sort_values(['submitterid', 'createdOn_diff'])
    submissions = submissions.groupby('submitterid', as_index=False).first()

    submissions['pearson_rank'] = submissions['pearson_correlation'].rank(ascending=False, method="min", na_option='bottom')
    submissions['cosine_rank'] = submissions['cosine'].rank(ascending=True, method="min", na_option='bottom')
    submissions['final_rank'] = (submissions['pearson_rank'] + submissions['cosine_rank']) / 2

    return submissions

def load_team_predictions(syn: synapseclient.Synapse, submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load prediction files for each team (one per team), return a DataFrame with columns:
    'stimulus', team1, team2, ..., teamN
    """
    team_dfs = []
    for _, row in submissions_df.iterrows():
        team_id = row['submitterid']
        sub_id = row['id']
        file_path = syn.getSubmission(sub_id)['filePath']
        df = pd.read_csv(file_path)
        df = df.sort_values(INDEX_COL).reset_index(drop=True)
        # Only keep feature columns and stimulus
        feature_cols = [col for col in df.columns if col != INDEX_COL]
        team_name = f"team_{team_id}"
        df = df[[INDEX_COL] + feature_cols]
        # Rename feature columns to team name (for stacking)
        df = df.rename(columns={col: team_name for col in feature_cols})
        team_dfs.append(df)
    # Merge all team columns on 'stimulus'
    merged_df = team_dfs[0][[INDEX_COL]]
    for df in team_dfs:
        merged_df = pd.merge(merged_df, df, on=INDEX_COL)
    return merged_df

def load_goldstandard(args) -> pd.DataFrame:
    """
    Load goldstandard file, sorted by stimulus.
    """
    gold_df = pd.read_csv(extract_gs_file(args.goldstandard_folder)).sort_values(INDEX_COL).reset_index(drop=True)
    return gold_df

def bootstrap_scores(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    n_bootstraps: int = 10000,
    sample_pct: float = 0.1,
    metric: str = "rmse"
) -> np.ndarray:
    """
    Bootstrap predictions and goldstandard, using 10% of the test data, for n_bootstraps iterations.
    Returns a matrix of shape (n_bootstraps, n_teams) with scores per team.
    """
    feature_cols = [col for col in pred_df.columns if col != INDEX_COL]
    n_samples = len(gold_df)
    n_teams = len(feature_cols)
    n_select = int(np.round(n_samples * sample_pct))
    scores = np.zeros((n_bootstraps, n_teams))

    gold_values = gold_df[feature_cols].values if set(feature_cols).issubset(gold_df.columns) else gold_df.drop(columns=[INDEX_COL]).values

    for team_idx, team in enumerate(feature_cols):
        pred_values = pred_df[team].values
        for i in range(n_bootstraps):
            idx = np.random.choice(n_samples, n_select, replace=True)
            gold_sample = gold_values[idx]
            pred_sample = pred_values[idx]
            if metric == "rmse":
                score = np.sqrt(mean_squared_error(gold_sample, pred_sample))
            elif metric == "pearson":
                score, _ = pearsonr(gold_sample, pred_sample)
            else:
                raise ValueError("Unsupported metric")
            scores[i, team_idx] = score
    return scores

def compute_bayes_factor(
    bootstrap_metric_matrix: np.ndarray,
    ref_pred_index: int,
    invert_bayes: bool = False
) -> np.ndarray:
    """
    Compute Bayes factors for a bootstrapped metric matrix.
    Args:
        bootstrap_metric_matrix: numpy array (bootstraps x teams)
        ref_pred_index: int, index of reference team/column
        invert_bayes: bool, whether to invert the Bayes factor
    Returns:
        numpy array of Bayes factors (length = number of teams)
    """
    M = bootstrap_metric_matrix - bootstrap_metric_matrix[:, [ref_pred_index]]
    K = []
    for col in range(M.shape[1]):
        x = M[:, col]
        num_ge_0 = np.sum(x >= 0)
        num_lt_0 = np.sum(x < 0)
        if num_lt_0 == 0:
            k = np.inf
        else:
            k = num_ge_0 / num_lt_0
        if num_ge_0 > num_lt_0:
            K.append(k)
        else:
            K.append(1 / k if k != 0 else np.inf)
    K[ref_pred_index] = 0
    K = np.array(K)
    if invert_bayes:
        with np.errstate(divide='ignore', invalid='ignore'):
            K = 1 / K
    return K

def plot_bootstrapped_rmse_and_bayes(
    boot_matrix: np.ndarray,
    bayes_factors: np.ndarray,
    team_names: list,
    output_prefix: str
) -> None:
    """
    Create RMSE boxplot and Bayes factor barplot similar to the R code.
    Args:
        boot_matrix: numpy array (n_bootstraps x n_teams), bootstrapped RMSE scores
        bayes_factors: numpy array (n_teams,), Bayes factors for each team
        team_names: list of team names (length n_teams)
        output_prefix: prefix for output files (svg/png)
    """
    # Prepare RMSE DataFrame for boxplot
    rmse_df = pd.DataFrame(boot_matrix, columns=team_names)
    rmse_long = rmse_df.melt(var_name="submission", value_name="bs_score")
    bayes_df = pd.DataFrame({"submission": team_names, "bayes": bayes_factors})

    # Merge Bayes factors
    rmse_long = rmse_long.merge(bayes_df, on="submission", how="left")

    # Categorize Bayes factors
    def bayes_cat(b):
        if b == 0:
            return "Top Performers"
        elif b <= 20:
            return "Bayes Factor ≤20"
        else:
            return "Bayes Factor >20"
    rmse_long["bayes_category"] = rmse_long["bayes"].apply(bayes_cat)
    bayes_df["bayes_category"] = bayes_df["bayes"].apply(lambda b: "Top Performer" if b == 0 else ("Bayes Factor ≤20" if b <= 20 else "Bayes Factor >20"))

    # RMSE boxplot
    plt.figure(figsize=(14, 10))
    order = rmse_long.groupby("submission")["bs_score"].mean().sort_values().index
    sns.boxplot(
        data=rmse_long,
        x="submission",
        y="bs_score",
        hue="bayes_category",
        order=order,
        linewidth=1.2,
        fliersize=1
    )
    plt.xticks(rotation=45, fontsize=19)
    plt.yticks(fontsize=18)
    plt.xlabel("Team", fontsize=16)
    plt.ylabel("Bootstrapped RMSE\n(num_iterations=10_000, random 10% sample)", fontsize=16)
    plt.legend(title=None, loc='upper left', fontsize=19)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.svg")
    plt.savefig(f"{output_prefix}.png")
    plt.close()

    # Bayes factor barplot
    plt.figure(figsize=(6, 10))
    order_bayes = bayes_df.sort_values("bayes")["submission"]
    sns.barplot(
        data=bayes_df,
        y="submission",
        x="bayes",
        hue="bayes_category",
        order=order_bayes,
        dodge=False
    )
    plt.axvline(x=3, linestyle="--", linewidth=1.2, color="black")
    plt.xlim(0, 20)
    plt.xlabel("Bayes Factor\n(Top Performer)", fontsize=16)
    plt.ylabel("")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bayes.svg")
    plt.savefig(f"{output_prefix}_bayes.png")
    plt.close()

def main():
    """Main function."""
    args = get_args()
    syn = synapseclient.Synapse()
    syn.login()

    # Load goldstandard data
    gold_df = load_goldstandard(args)

    # Select final round submissions
    subview_id = args.subview_id if args.subview_id else SUBMISSION_VIEWS[f"Task {args.task}"]
    submissions_df = select_final_round_submissions(syn, subview_id, args.evaluation_id)

    # Load team predictions
    pred_df = load_team_predictions(syn, submissions_df)

    # Bootstrap scores
    boot_matrix = bootstrap_scores(
        pred_df,
        gold_df,
        n_bootstraps=args.n_bootstraps,
        sample_pct=args.sample_pct,
        metric=args.metric
    )

    # Compute Bayes factors
    ref_pred_index = 0  # Assuming the first team is the reference team
    bayes_factors = compute_bayes_factor(boot_matrix, ref_pred_index, invert_bayes=False)

    # Prepare team names
    team_names = [f"Team {i+1}" for i in range(boot_matrix.shape[1])]

    # Plot results
    plot_bootstrapped_rmse_and_bayes(scores_matrix, bayes_factors, team_names, args.output_prefix)

if __name__ == "__main__":
    main()
