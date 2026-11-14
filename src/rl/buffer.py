from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    sample_id: str
    action: int
    components: dict


class RolloutBuffer:
    def __init__(self):
        self.transitions = []

    def append(self, sample_id, action, components):
        self.transitions.append(Transition(sample_id, action, dict(components)))

    def drain(self):
        result, self.transitions = self.transitions, []
        return result
