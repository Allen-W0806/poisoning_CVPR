param([Parameter(Mandatory=$true)][string]$InputJson)
$ErrorActionPreference='Stop'
$reviewInput=(Resolve-Path -LiteralPath $InputJson).Path
$reviewValue=Get-Content -LiteralPath $reviewInput -Raw -Encoding utf8 | ConvertFrom-Json
if ($reviewValue.schema_version -ne 'independent_edit_review_v1') { throw 'Unsupported review format; export from reviews/workflow/index.html' }
if (-not $reviewValue.provenance.reviewer_id -or -not $reviewValue.provenance.review_session_id) { throw 'Reviewer/session provenance missing' }
$reviewToken=[guid]::NewGuid().ToString('N')
$reviewRoot='/storage/home/ydk297/projects/poisoning/reviews/incoming'
& ssh utc-login "mkdir -p $reviewRoot"
if ($LASTEXITCODE -ne 0) { throw 'Cannot prepare own-cluster review inbox' }
& scp $reviewInput ("utc-login:"+$reviewRoot+'/.upload_'+$reviewToken)
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; no complete JSON submission was published' }
& ssh utc-login ("mv -n "+$reviewRoot+'/.upload_'+$reviewToken+' '+$reviewRoot+'/review_'+$reviewToken+'.json')
if ($LASTEXITCODE -ne 0) { throw 'Atomic submission publication failed' }
Write-Output 'Review submitted with a unique filename. The existing monitor validates metadata and records agreement; human verification and scientific reward approval remain separate.'
