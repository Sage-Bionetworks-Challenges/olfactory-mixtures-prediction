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
import numpy as np
import pandas as pd  
import synapseclient


SUBMISSION_VIEWS = {
    "Task 1": "syn66279193",
    "Task 2": "syn66484079"
}

evaluation_id = "Final Round DREAM Olfactory Mixtures Prediction Challenge 2025 - Task 1"

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