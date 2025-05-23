"""Update Submission Ranks.

This script will rank submissions made to the DREAM Olfactory Mixtures 
Prediction Challenge 2025 - (syn64743570) according to 
Pearson Correlation and Cosine.
"""
from challengeutils.annotations import update_submission_status
from challengeutils.utils import update_single_submission_status
import synapseclient


SUBMISSION_VIEWS = {
    "Task 1": "syn66279193",
    "Task 2": "syn66484079"
}


def rank_submissions(syn, subview_id):
    """
    Get scored submissions. Compute ranks for both metrics.
    Outputs a final averaged rank and sorted leaderboard.
    """
    query = (f"SELECT id, pearson_correlation, cosine, createdOn, submitterid FROM {subview_id} "
             f"WHERE score_status = 'SCORED' ")
    submissions = syn.tableQuery(query).asDataFrame()
    submissions['pearson_rank'] = submissions['pearson'].rank(ascending=False, method="min")
    submissions['cosine_rank'] = submissions['cosine'].rank(ascending=True, method="min")
    submissions['current_rank'] = (submissions['pearson_rank'] + submissions['cosine_rank']) / 2
    submissions = submissions.sort_values(
        by=['pearson_correlation', 'cosine'],
        ascending=False
    )

    return submissions


def annotate_submissions(syn, sub_df):
    """Annotate submissions with their new rank."""
    for _, row in sub_df.iterrows():
        annots = {'current_rank': int(row['rank'])}
        sub_status = syn.getSubmissionStatus(row['id'])
        updated = update_single_submission_status(
            sub_status, annots, is_private=False)
        updated = update_submission_status(updated, annots)
        syn.store(updated)


def main():
    """Main function."""
    syn = synapseclient.Synapse()
    syn.login(silent=True)

    for task, syn_id in SUBMISSION_VIEWS.items():
        ranked_subs = rank_submissions(syn, syn_id)
        if not ranked_subs.empty:
            annotate_submissions(syn, ranked_subs)
        print(f"Annotating {task} submissions DONE ✓")


if __name__ == "__main__":
    main()
