"""Refresh current review metadata and dashboard without rerunning completed stages."""
import json
import _bootstrap
from src.progression.review_workflow import refresh_review_analysis
from src.progression.dashboard import refresh
if __name__=='__main__':
    reviews=refresh_review_analysis(_bootstrap.ROOT)
    print(json.dumps({'reviews':{'submission_files':reviews.get('submission_files',0),'independent_pairs':reviews.get('independently_reviewed_consensus_pairs',0)},
       'dashboard':refresh(_bootstrap.ROOT)}))
