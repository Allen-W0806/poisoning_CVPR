#!/usr/bin/env bash
# Model/data provenance and non-destructive setup for the verified clean cohort.
# Original Windows source mirror: H:/Project/Poisoning
# Original Linux project: /storage/home/ydk297/projects/poisoning
# Shared HF cache: /storage/home/ydk297/.cache/huggingface
# Qwen/Qwen3-VL-2B-Instruct @ 89644892e4d85e24eaac8bacfd4f463576704203
# timbrooks/instruct-pix2pix @ 31519b5cb02a7fd89b906d88731cd4d6a7bbf88d
# openai/clip-vit-large-patch14 @ 32bd64288804d66eefd0ccbe215aa642df71cc41
# Qwen snapshot: /storage/home/ydk297/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
# IP2Pix snapshot: /storage/home/ydk297/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix/snapshots/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d
# CLIP snapshot: /storage/home/ydk297/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41
# VQAv2 questions: /storage/home/ydk297/projects/poisoning/data/raw/vqav2/v2_Questions_Train_mscoco.zip
# VQAv2 annotations: /storage/home/ydk297/projects/poisoning/data/raw/vqav2/v2_Annotations_Train_mscoco.zip
# Fixed COCO cohort: /storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/images
# Metric trunk: /storage/home/ydk297/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
# VQAv2 archives: https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/
# COCO originals: https://s3.amazonaws.com/images.cocodataset.org/train2014/
set -euo pipefail
MODE="${1:---check}"
if [[ "$MODE" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/download_and_setup.sh [--check|--download-missing|--paths-json]
Default --check performs CPU/filesystem checks only; it downloads nothing.
--download-missing is an explicit Linux-only opt-in and may download ~8.7 GB.
Existing models, image bytes, protocol and experiment splits are never replaced.
Override PROJECT_ROOT, HF_HOME, TORCH_HOME, PYTHON_BIN or MANIFEST_PATH as needed.
See reports/ASSET_LOCATIONS_20261114.json for exact paths, revisions and sources.
HELP
  exit 0
fi
case "$MODE" in --check|--download-missing|--paths-json) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
SCRIPT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-$SCRIPT_REPO_ROOT}"
export PROJECT_ROOT="${PROJECT_ROOT:-/storage/home/ydk297/projects/poisoning}"
export HF_HOME="${HF_HOME:-/storage/home/ydk297/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/storage/home/ydk297/.cache/torch}"
export ASSET_CATALOG="${ASSET_CATALOG:-$REPO_ROOT/reports/ASSET_LOCATIONS_20261114.json}"
export MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/outputs/overnight_benign_20261112T051015Z/remote/manifest.json}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv-editor/bin/python" ]]; then PYTHON_BIN="$PROJECT_ROOT/.venv-editor/bin/python"; else PYTHON_BIN=python3; fi
fi
"$PYTHON_BIN" -B - "$MODE" <<'PY'
import hashlib,json,os,shutil,sys,tempfile,urllib.request
from pathlib import Path
from datetime import datetime,timezone

mode=sys.argv[1];catalog=json.loads(Path(os.environ['ASSET_CATALOG']).read_text(encoding='utf-8'))
project=Path(os.environ['PROJECT_ROOT']).resolve();hf=Path(os.environ['HF_HOME']).resolve();torch=Path(os.environ['TORCH_HOME']).resolve()
if mode=='--paths-json':print(json.dumps(catalog,indent=2));raise SystemExit(0)
manifest_path=Path(os.environ['MANIFEST_PATH']);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(manifest_path)!=catalog['clean_cohort']['manifest_SHA256']:raise ValueError('Fixed cohort manifest SHA mismatch')
models=[];files=[]
for item in catalog['models']:
    folder=hf/'hub'/('models--'+item['repo_id'].replace('/','--'))/'snapshots'/item['revision']
    missing=[f for f in item['files'] if not (folder/f['relative']).is_file() or (folder/f['relative']).stat().st_size!=f['bytes']]
    models.append((item,folder,missing))
for item in catalog['dataset_archives']:
    files.append((project/'data/raw/vqav2'/Path(item['absolute_path']).name,item['source_URL'],item['SHA256'],item['bytes']))
for rows in manifest['splits'].values():
    for row in rows:
        target=(project/row['image_path']).resolve()
        if not target.is_relative_to(project):raise ValueError('Unsafe manifest path')
        files.append((target,row['URL'],row['image_sha256'],None))
for item in catalog['metric_assets']:
    target=torch/'hub/checkpoints'/Path(item['absolute_path']).name if item['kind']=='torch_trunk' else project/item['project_relative_path']
    files.append((target,item['source_URL'],item['SHA256'],item['bytes']))
missing_files=[r for r in files if not r[0].is_file()]
missing_model_bytes=sum(f['bytes'] for _,_,missing in models for f in missing)
estimated=missing_model_bytes+sum(r[3] or 300000 for r in missing_files)
if mode=='--download-missing':
    if os.name!='posix':raise SystemExit('Download mode targets Linux. Run on the cluster; Windows holds only the source mirror.')
    def available(path):
        path=Path(path)
        while not path.exists():path=path.parent
        return shutil.disk_usage(path).free
    required=int(estimated*1.25)+100000000
    if min(available(project),available(hf),available(torch))<required:raise RuntimeError('Insufficient free disk for bounded missing-asset download')
    print(json.dumps({'planned_missing_bytes':estimated,'disk_reservation_with_margin':required}),flush=True)
    # Base weights stay in the shared cache; no model is instantiated.
    if missing_model_bytes:
        from huggingface_hub import snapshot_download
        for item,folder,missing in models:
            if missing:
                snapshot_download(repo_id=item['repo_id'],revision=item['revision'],cache_dir=str(hf/'hub'),allow_patterns=[f['relative'] for f in missing])
    scratch_parent=project/'tmp';scratch_parent.mkdir(parents=True,exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='asset_setup_',dir=scratch_parent)).resolve()
    try:
        for index,(target,url,digest,_) in enumerate(missing_files):
            temporary=scratch/(str(index)+'.partial')
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as stream:shutil.copyfileobj(response,stream)
            if sha(temporary)!=digest:raise ValueError('Downloaded asset SHA mismatch; existing data untouched')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('Asset appeared concurrently; refusing overwrite')
            os.replace(temporary,target)
    finally:
        checked=scratch.resolve()
        if checked.parent!=scratch_parent.resolve() or not checked.name.startswith('asset_setup_'):raise ValueError('Scratch cleanup scope mismatch')
        shutil.rmtree(checked)
errors=[]
for item,folder,_ in models:
    for record in item['files']:
        path=folder/record['relative']
        if not path.is_file() or path.stat().st_size!=record['bytes']:errors.append(str(path))
for target,_,digest,_ in files:
    if not target.is_file() or sha(target)!=digest:errors.append(str(target))
print(json.dumps({'status':'PASS_EXISTING_ASSET_CHECK' if not errors else 'FAIL_MISSING_OR_CHANGED_ASSETS','checked_UTC':datetime.now(timezone.utc).isoformat(),
    'mode':mode,'model_snapshots':len(models),'fixed_source_images':sum(len(r) for r in manifest['splits'].values()),
    'files_SHA256_checked':len(files),'missing_or_changed':errors,'new_GPU_jobs':0,'downloaded_in_check_mode':False},indent=2))
raise SystemExit(1 if errors else 0)
PY
