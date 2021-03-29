from typing import List


class Metrics():
    def __init__(self, fmt):
        self.fmt = fmt


# class DispactchedMetric:
#     def __init__(self,history:List,steps:List):
#         self.history = history
#         self.steps = steps

class LossTracker(Metrics):
    """Track loss for entire train period and validation period
    """

    def __init__(self, loss_type: str, fmt=":0.2f"):
        super().__init__(fmt)
        self.loss_type = loss_type
        self.history = []
        self.steps = []  # Mostly intended for epoch_num or batch_idx

    def reset(self):
        self.history = []
        self.steps = []

    def update(self, val, step, multiple=1):
        self.history.append(self.avg)
        self.steps.append(step)

    def asdict(self):
        return self.__dict__
