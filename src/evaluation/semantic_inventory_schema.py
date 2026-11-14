"""Schema v2: preserve model outputs; normalize explicit integer counts only."""
import json
from .semantic_review import FIELDS

def parse_inventory_v2(raw):
    try:
        value=json.loads(raw[raw.find('{'):raw.rfind('}')+1])
        count=value.get('count')
        if isinstance(count,int) and not isinstance(count,bool) and count>=0:
            value['count']=str(count)
        if not all(isinstance(value.get(k),str) and value[k].strip() for k in FIELDS):
            raise ValueError('Missing/invalid inventory descriptions')
        return {**value,'parse_ok':True,'raw_output':raw,'schema_version':'v2_integer_count_representation'}
    except (ValueError,TypeError,AttributeError):
        return {'parse_ok':False,'raw_output':raw,'reason':'Unparseable inventory; no inferred fields',
            'schema_version':'v2_integer_count_representation'}
