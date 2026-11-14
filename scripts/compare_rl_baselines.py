"""Prepare/inspect comparison contracts only; never silently launches a sweep."""
import json
import yaml
import _bootstrap

if __name__=="__main__":
    c=yaml.safe_load((_bootstrap.ROOT/"configs/phase3_comparison.yaml").read_text())
    assert c["status"]=="PREPARED_NOT_RUN" and c["final_test_access"]=="prohibited_during_selection"
    assert set(c["methods"])=={"M0","M1","M2","M3","M4"}
    print(json.dumps(c,indent=2))
