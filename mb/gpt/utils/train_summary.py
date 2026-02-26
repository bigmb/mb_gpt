## summary file
from mb.utils.logging import logg

__all__ = ['TrainSummary']

class TrainSummary:
    def __init__(self, data: dict,logger=None):
        self.summary = data
        self.output_path = self.summary.get('output_path', './summary_output.json')
        self.print_output = self.summary.get('print_output', False)
        self.logger = logger

    def __repr__(self):
        return f"TrainSummary(TrainSummary={self.summary})"
    
    def _epoch_data(self, epoch: int, loss: float, **kwargs):
        pass

    def _loss_data(self, loss: float, **kwargs):
        pass

    def _extra_data(self, **kwargs):
        pass

    def log_epoch(self, epoch: int, loss: float, **kwargs):
        pass