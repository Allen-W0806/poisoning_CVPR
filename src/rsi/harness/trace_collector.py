"""Append-only execution histories with measured resources, no hidden test labels."""
from dataclasses import dataclass, asdict
import json
import math
import time
from pathlib import Path

@dataclass
class Resources:
    wall_seconds: float = 0.
    cpu_seconds: float = 0.
    diagnostic_operations: int = 0
    bytes_read: int = 0
    model_calls: int = 0
    gpu_seconds: float = 0.

    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid resource field: '+key)
        return self

    def add(self,other):
        other.validate()
        for key,value in asdict(other).items():setattr(self,key,getattr(self,key)+value)
        return self.validate()

class TraceCollector:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.touch(exist_ok=False)
        self.started=time.perf_counter();self.cpu=time.process_time();self.count=0

    def append(self,event,payload,split='train'):
        if split!='train':raise ValueError('Editor trace collector accepts training-side cases only')
        record={'event':event,'sequence':self.count,'split':split,'payload':payload}
        encoded=json.dumps(record,allow_nan=False,sort_keys=True)
        with self.path.open('a',encoding='utf8') as f:f.write(encoded+'\n')
        self.count+=1
        return record

    def resources(self):
        return Resources(wall_seconds=time.perf_counter()-self.started,cpu_seconds=time.process_time()-self.cpu).validate()
