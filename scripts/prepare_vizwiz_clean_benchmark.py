"""Stage the official8000-image test benchmark with newly public2026 answers."""
import argparse
import hashlib
import json
import shutil
import stat
import sys
import time
import urllib.request
import zipfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import sha256,write_json

IMAGE_URL="https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
ANSWER_URL="https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json"
ARCHIVE_BYTES=3975272799
ANSWER_BYTES=8227183


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--campaign",required=True,type=Path)
    p.add_argument("--resume-staging",action="store_true");args=p.parse_args()
    c=args.campaign.resolve()
    if not c.is_relative_to(ROOT/"outputs"):raise ValueError("Unsafe campaign path")
    if shutil.disk_usage(ROOT).free<12_000_000_000:raise RuntimeError("Need12GB verified cluster staging margin")
    out=c/"vizwiz";out.mkdir(exist_ok=args.resume_staging)
    dest=ROOT/"data/external/vizwiz_test_release20260428";dest.mkdir(parents=True,exist_ok=args.resume_staging)
    if args.resume_staging:
        previous=json.loads((out/"data_receipt.json").read_text())
        if previous['status']!='FAILED' or not previous.get('failure','').startswith('ValueError: Question versions disagree'):
            raise ValueError('Only the diagnosed case-format verification failure can resume staging')
        shutil.copy2(out/'data_receipt.json',out/'data_receipt_initial_verification_failure.json')
    key=hashlib.sha256(c.name.encode()).hexdigest()[:10]
    scratch=(ROOT/"tmp"/("rd_viz_"+key)).resolve();scratch.mkdir(exist_ok=False)
    receipt={"status":"STAGING","images_url":IMAGE_URL,"annotations_url":ANSWER_URL,
        "public_release":"official site April2026 self-evaluation release, fetched before any new benchmark outputs",
        "expected_archive_bytes":ARCHIVE_BYTES,"estimated_peak_staging_bytes":9_000_000_000,
        "C_drive_bytes":0,"free_before":shutil.disk_usage(ROOT).free,"scratch":str(scratch),"data_root":str(dest)}
    write_json(out/"data_receipt.json",receipt,exclusive=not args.resume_staging)
    try:
        raw=urllib.request.urlopen(ANSWER_URL,timeout=60).read()
        if len(raw)!=ANSWER_BYTES:raise ValueError("Official answer file size changed")
        annotations=json.loads(raw)
        if len(annotations)!=8000 or any(len(row["answers"])!=10 for row in annotations):
            raise ValueError("8000 questions and original10 human references required")
        (dest/"VQA_test.json").write_bytes(raw)
        receipt["annotation_SHA256"]=sha256(dest/"VQA_test.json")
        question_source=c/"external_preparation/vizwiz_llava_test_original.jsonl"
        questions=[json.loads(line) for line in question_source.read_text().splitlines()]
        if len(questions)!=8000 or len({r['question_id'] for r in questions})!=8000:
            raise ValueError("Original LLaVA question list is incomplete/duplicated")
        refs={row["image"]:row for row in annotations}
        if len(refs)!=8000 or {row['image'] for row in questions}!=set(refs):raise ValueError("Image-ID mismatch")
        for question in questions:
            name=question['image']
            if PurePosixPath(name).name!=name:raise ValueError("Unsafe annotation image path")
            source_question=question['text'].split("\nWhen the provided information")[0].strip()
            if source_question.casefold()!=refs[name]['question'].strip().casefold():
                raise ValueError("Question versions disagree beyond capitalization")
        receipt['question_verification']='All8000 questions match official2026 references after capitalization-only comparison; actual prompts/references are unchanged'
        shutil.copy2(question_source,dest/"llava_test.jsonl")
        receipt['question_SHA256']=sha256(dest/'llava_test.jsonl')
        archive=scratch/'test.zip';started=time.monotonic();downloaded=0;next_notice=500_000_000
        with urllib.request.urlopen(IMAGE_URL,timeout=90) as response,archive.open('xb') as f:
            while True:
                block=response.read(8*1024*1024)
                if not block:break
                downloaded+=len(block)
                if downloaded>ARCHIVE_BYTES:raise ValueError("Image archive larger than declared")
                f.write(block)
                if downloaded>=next_notice:
                    print(json.dumps({'downloaded_bytes':downloaded,'expected':ARCHIVE_BYTES}),flush=True);next_notice+=500_000_000
        if downloaded!=ARCHIVE_BYTES:raise ValueError("Incomplete image archive")
        receipt['archive_SHA256']=sha256(archive)
        with zipfile.ZipFile(archive) as z:
            if sum(info.file_size for info in z.infolist())>8_000_000_000:raise ValueError("Unexpected extracted size")
            for info in z.infolist():
                part=PurePosixPath(info.filename)
                if part.is_absolute() or '..' in part.parts or stat.S_ISLNK(info.external_attr>>16):raise ValueError("Unsafe archive member")
                if not (dest/info.filename).resolve().is_relative_to(dest.resolve()):raise ValueError("Archive escape")
            z.extractall(dest)
        candidates=[folder for folder in [dest,*dest.iterdir()] if folder.is_dir() and (folder/questions[0]['image']).is_file()]
        if len(candidates)!=1:raise ValueError("Cannot resolve image root")
        image_root=candidates[0]
        from PIL import Image
        inventory=[]
        for row in questions:
            path=image_root/row['image']
            with Image.open(path) as image:image.verify()
            inventory.append({'image':row['image'],'question_id':row['question_id'],'bytes':path.stat().st_size,'SHA256':sha256(path)})
        if len(inventory)!=8000:raise ValueError("Incomplete decoded image coverage")
        write_json(out/'image_inventory.json',inventory,exclusive=True)
        receipt.update(status='READY',image_root=str(image_root),images=8000,questions=8000,
            all_images_decoded=True,image_bytes=sum(r['bytes'] for r in inventory),
            full_test_not_original_testdev_subset=True,download_and_staging_seconds=time.monotonic()-started)
        archive.unlink()
    except BaseException as exc:
        receipt.update(status='FAILED',failure=f'{type(exc).__name__}: {exc}');raise
    finally:
        receipt['free_after']=shutil.disk_usage(ROOT).free;write_json(out/'data_receipt.json',receipt)
        target=scratch.resolve()
        if not target.is_relative_to((ROOT/'tmp').resolve()) or target.name!='rd_viz_'+key:raise RuntimeError('Unsafe scratch cleanup')
        shutil.rmtree(target)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
