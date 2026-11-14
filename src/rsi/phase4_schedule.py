"""Limited validation-driven schedule extension of the existing RSI controller.

Accepting an exploratory schedule revision is not accepting a quality gain.
The legacy candidate-gain selector remains unchanged for historical experiments.
"""
import copy
import math
from .controller import RSIController
from .revision import validate_revision
from ..evaluation.phase4_statistics import rsi_plateau
from ..utils.logging import utcnow

class Phase4ScheduleController(RSIController):
    def revise_schedule(self,looks,validation_score,cumulative_cost_gpu_hours,validation_hash):
        if validation_hash!=self.state['validation_hash']:raise ValueError('Validation identity changed')
        if not math.isfinite(validation_score) or not math.isfinite(cumulative_cost_gpu_hours):raise ValueError('Nonfinite schedule input')
        if cumulative_cost_gpu_hours<self.state['cost_gpu_hours']:raise ValueError('Cumulative cost decreased')
        eligible=rsi_plateau(looks) and self.state['config_version']==0
        within=cumulative_cost_gpu_hours<=self.state['budget_gpu_hours']
        accept=eligible and within
        proposed=validate_revision(self.state['training_config'],{'learning_rate':5e-6})
        revision={'revision_number':len(self.state['history'])+1,'timestamp':utcnow(),
            'previous_config':copy.deepcopy(self.state['training_config']),'proposed_config':proposed,
            'validation_hash':validation_hash,'validation_score':validation_score,
            'cost_gpu_hours':cumulative_cost_gpu_hours-self.state['cost_gpu_hours'],
            'cumulative_cost_gpu_hours':cumulative_cost_gpu_hours,'accepted_schedule':accept,
            'quality_improvement_accepted':False,'reason':'two unresolved validation gains' if accept else ('budget_exceeded' if not within else 'insufficient plateau evidence'),
            'evidence':copy.deepcopy(looks[-2:]),'policy':'single exploratory LR cooldown; no candidate quality-gain or rollback claim'}
        self.memory.append(revision);self.state['history'].append(revision)
        self.state['cost_gpu_hours']=cumulative_cost_gpu_hours
        self.state['validation_score']=validation_score
        if accept:self.state['training_config']=proposed;self.state['config_version']+=1
        self.memory.snapshot(self.state)
        return revision
