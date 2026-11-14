"""Versioned human-only review ingestion with explicit visible-text preservation."""
import datetime

ATTRS=('scene_preserved','objects_preserved','identity_count_preserved','color_spatial_preserved',
       'text_preserved','reference_answer_valid','instruction_followed')


def validate_row(row,case):
    # Preserve original export bytes in the caller; normalize the legacy UI
    # synonym only in the parsed comparison object.
    row=dict(row)
    for k in ATTRS:
        if row.get(k)=='ambiguous':row[k]='uncertain'
    required={'reviewer_id','independence_attestation','reviewed_utc','source_image_id','question_id',
              'original_sha256','edited_sha256','semantic_label','confidence','failure_reason','notes',*ATTRS}
    if not required<=set(row):raise ValueError('Incomplete v3 human review')
    if not row['reviewer_id'].strip() or row['independence_attestation']!='yes':raise ValueError('Independent human attestation required')
    if str(case['source_image_id']).isdigit() and row['reviewer_id'].upper().startswith(('SYNTHETIC','AI_','MODEL_')):raise ValueError('Fixture/model identifiers cannot certify research images')
    timestamp=datetime.datetime.fromisoformat(row['reviewed_utc'].replace('Z','+00:00'))
    if timestamp.tzinfo is None:raise ValueError('Timezone required')
    for k in ['source_image_id','question_id','original_sha256','edited_sha256']:
        if str(row[k])!=str(case[k]):raise ValueError('Review/image association mismatch: '+k)
    if any(row[k] not in {'yes','no','uncertain','not_applicable'} for k in ATTRS):raise ValueError('Missing/ambiguous review choice')
    if row['semantic_label'] not in {'valid','invalid','ambiguous'} or row['confidence'] not in {'low','medium','high'}:raise ValueError('Review label/confidence required')
    if row['semantic_label']!='valid' and not row['failure_reason'].strip():raise ValueError('Invalid/ambiguous label needs reason')
    preserve=ATTRS[:-1]
    if row['semantic_label']=='valid' and (row['reference_answer_valid']!='yes' or any(row[k] not in {'yes','not_applicable'} for k in preserve)):
        raise ValueError('Valid label conflicts with preservation judgment')
    return dict(row)


def consensus(rows,case,adjudication=None):
    checked=[validate_row(r,case) for r in rows]
    ids=[r['reviewer_id'] for r in checked]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate reviewer is not independent')
    if len(checked)<2:return dict(status='PENDING_TWO_HUMANS',semantic_label=None,instruction_success=None)
    # Any disagreement on the dimensions needs a distinct human adjudicator.
    agreed=all(all(r[k]==checked[0][k] for k in (*ATTRS,'semantic_label')) for r in checked[1:])
    if not agreed:
        if adjudication is None:return dict(status='NEEDS_HUMAN_ADJUDICATION',semantic_label=None,instruction_success=None)
        final=validate_row(adjudication,case)
        if final['reviewer_id'] in ids:raise ValueError('Adjudicator must be a third independent human')
    else:final=checked[0]
    return dict(status='HUMAN_REVIEWED',semantic_label=final['semantic_label'],instruction_success=final['instruction_followed'],reviewer_ids=ids)
