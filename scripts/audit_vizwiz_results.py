"""CPU recomputation of complete VizWiz raw-response scoring, without a VLM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import sha256, write_json
from src.evaluation.vizwiz_official import score_answer, PROTOCOL, reference_metadata
from scripts.evaluate_shadowcast_vizwiz import preflight


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--campaign",type=Path,required=True)
    c=p.parse_args().campaign.resolve()
    if not c.is_relative_to(ROOT/"outputs"):raise ValueError("Expected canonical campaign")
    _,receipt,questions=preflight(c)
    references={r['image']:r for r in json.loads((Path(receipt['data_root'])/'VQA_test.json').read_text())}
    inventory={r['image']:r for r in json.loads((c/'vizwiz/image_inventory.json').read_text())}
    checks={}
    for n in [0,50]:
        raw_path=c/f'vizwiz/predictions_{n}.jsonl';score_path=c/f'vizwiz/scores_{n}.jsonl'
        predictions=[json.loads(l) for l in raw_path.read_text().splitlines()]
        scores=[json.loads(l) for l in score_path.read_text().splitlines()]
        summary=json.loads((c/f'vizwiz/summary_{n}.json').read_text())
        if len(predictions)!=8000 or len(scores)!=8000:raise ValueError('Never audit a reduced denominator')
        total=0.
        for question, prediction, stored in zip(questions,predictions,scores):
            if prediction['question_id']!=question['question_id'] or stored['question_id']!=question['question_id']:
                raise ValueError('Question pairing changed')
            if prediction['prompt']!=question['text'] or stored['image']!=question['image']:
                raise ValueError('Original question or image changed')
            if stored['image_SHA256']!=inventory[question['image']]['SHA256']:
                raise ValueError('Source inventory mismatch')
            annotation=references[question['image']]
            actual=score_answer(prediction['text'],{'image':question['image'],
                'answer_type':annotation['answer_type'],'answers':[r['answer'] for r in annotation['answers']]})
            for key in ['raw_answer','normalized_answer','human_reference_answers','scoring_protocol']:
                if stored[key]!=actual[key]:raise ValueError('Stored scoring evidence changed: '+key)
            if abs(stored['score']-actual['score'])>1e-12:raise ValueError('Official per-question score does not reproduce')
            total+=actual['score']
        if summary['questions']!=8000 or summary['dropped_cases']!=0:
            raise ValueError('Summary denominator changed')
        if abs(total/8000-summary['clean_VizWiz_full_test_accuracy'])>1e-12:
            raise ValueError('Summary accuracy does not reproduce')
        if sha256(raw_path)!=summary['raw_predictions_SHA256'] or sha256(score_path)!=summary['scores_SHA256']:
            raise ValueError('Raw response or score-file digest mismatch')
        checks[str(n)]={'all8000_raw_answers_rescored':True,'mean_accuracy':total/8000,
            'raw_predictions_SHA256':summary['raw_predictions_SHA256'],'scores_SHA256':summary['scores_SHA256']}
    result={'status':'PASS','UTC':datetime.now(timezone.utc).isoformat(),'conditions':checks,
            'scoring_protocol':PROTOCOL,'official_reference':reference_metadata(),
            'annotation_SHA256':receipt['annotation_SHA256'],'question_SHA256':receipt['question_SHA256'],
            'new_model_calls':0,'scope':'deterministic raw-score recomputation, not a full independent scientific integrity audit'}
    write_json(c/'vizwiz/raw_score_audit.json',result,exclusive=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
