"""Bootstrap Submission Scores by Task.

This script will bootstrap submissions by task made to the DREAM Olfactory Mixtures
Prediction Challenge 2025 - (syn64743570) according to
Pearson Correlation and Cosine. Only final round submissions are considered.
Only the latest submission for each submitter is kept.
A team was given perission to have a submission that is not the latest included.
"""
from challengeutils.annotations import update_submission_status
from challengeutils.utils import update_single_submission_status
from datetime import datetime, timezone
# Merge all team prediction columns together may be eliminated if no longer needed
from functools import reduce
import numpy as np
import pandas as pd  
import synapseclient


SUBMISSION_VIEWS = {
    "Task 1": "syn66279193",
    "Task 2": "syn66484079"
}

INDEX_COL = "stimulus"

evaluation_id = "Final Round DREAM Olfactory Mixtures Prediction Challenge 2025 - Task 1"

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

def select_final_round_submissions(syn, subview_id, evaluation_id):
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
    )
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

def compute_bayes_factor(bootstrap_metric_matrix, ref_pred_index, invert_bayes=False):
    """
    Compute Bayes factors for a bootstrapped metric matrix.
    Args:
        bootstrap_metric_matrix: numpy array (bootstraps x teams)
        ref_pred_index: int, index of reference team/column
        invert_bayes: bool, whether to invert the Bayes factor
    Returns:
        numpy array of Bayes factors (length = number of teams)
    """
    # Difference from reference column
    M = bootstrap_metric_matrix - bootstrap_metric_matrix[:, [ref_pred_index]]
    K = []
    for col in range(M.shape[1]):
        x = M[:, col]
        num_ge_0 = np.sum(x >= 0)
        num_lt_0 = np.sum(x < 0)
        if num_lt_0 == 0:
            k = np.inf  # Avoid division by zero
        else:
            k = num_ge_0 / num_lt_0
        # Logic handles whether reference column is the best set of predictions.
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


def build_bootstrap_submissions(syn, query_df, gold_df, composite_key, pred_col="Predicted_Experimental_Values", gold_col="Experimental_Values"):
    """
    Reads in prediction files and combines them with the goldstandard into a single DataFrame for bootstrapping.
    Args:
        syn: Synapse client
        query_df: DataFrame with submission IDs and team names
        gold_df: Goldstandard DataFrame
        composite_key: List of columns to join on (e.g., ["Dataset", "Mixture_1", "Mixture_2"])
        pred_col: Name of the prediction column in submission files
        gold_col: Name of the goldstandard column
    Returns:
        DataFrame with predictions from all teams and goldstandard values
    """

    # Create a dataframe for the gold standard
    gold_df = pd.read_csv(gold).sort_values(INDEX_COL).reset_index(drop=True)
    expected_cols = (
        gold_df.dtypes.to_dict()
        )

    # Create a dataframe for the predictions
    pred_df = pd.read_csv(
        pred,
        usecols=expected_cols,
        dtype=expected_cols,
        float_precision="round_trip",
        ).sort_values(INDEX_COL).reset_index(drop=True)

    # Confirm the new dataframes have the same number of rows
    if len(pred_df) == len(gold_df):
        feature_cols = pred_df.columns.difference(["stimulus"])

        for i in range(len(pred_df)):
            pred_vector = pred_df.loc[i, feature_cols].values
            true_vector = gold_df.loc[i, feature_cols].values
    
    # Commenting out for now to see if some of the existing scoring logic can be used instead
    # Get file paths for each submission
    # pred_filenames = {row['submitterid']: syn.getSubmission(row['id'])['filePath'] for _, row in query_df.iterrows()}

    # Read and format each team's predictions
    # team_dfs = []
    # for team, file_path in pred_filenames.items():
    #     df = pd.read_csv(file_path)
    #     df = df[composite_key + [pred_col]].copy()
    #     df.rename(columns={pred_col: team}, inplace=True)
    #     team_dfs.append(df)

    # Merge all team prediction columns together
    # from functools import reduce
    # submissions_df = reduce(lambda left, right: pd.merge(left, right, on=composite_key, how='left'), team_dfs)

    # Ensure row order matches goldstandard
    # submissions_df = submissions_df.set_index(composite_key)
    # gold_df = gold_df.set_index(composite_key)
    # submissions_df = submissions_df.loc[gold_df.index].reset_index()

    # Merge in goldstandard target values
    submissions_df = pd.merge(submissions_df, gold_df.reset_index(), on=composite_key, how='left')

    # Rename goldstandard column to "gold"
    submissions_df.rename(columns={gold_col: "gold"}, inplace=True)

    return submissions_df

# Example build_bootstrap_submissions usage:
# composite_key = ["Dataset", "Mixture_1", "Mixture_2"]
# query_df should have columns: 'id' and 'submitterid'
# submissions_df = build_bootstrap_submissions(syn, query_df, gold, composite_key)
# print(submissions_df.head())

