"""Update Submission Ranks.

This script will rank submissions made to the Preterm Birth Prediction -
Microbiome Challenge (syn26133770) according to AUC-ROC and AUPRC.
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
    Get scored submissions and rank them according to AUC-ROC, followed
    by AUPRC for tie-breakers.
    """
    query = (f"SELECT id, pearson_correlation, cosine, createdOn, submitterid FROM {subview_id} "
             f"WHERE submission_status = 'SCORED' "
             f"AND status = 'SCORED'")
    submissions = syn.tableQuery(query).asDataFrame()
    submissions['rank'] = (
        submissions[['pearson_correlation', 'cosine']]
        .apply(tuple, axis=1)
        .rank(method='min', ascending=False)
        .astype(int)
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