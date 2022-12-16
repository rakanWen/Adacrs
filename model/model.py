from abc import abstractmethod
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    @abstractmethod
    def fusion_output_size(self):
        """返回融合表征经过最终线性层之前的维度"""

    @abstractmethod
    def get_fusion_output(self, **kwargs):
        """返回融合表征经过最终线性层的输入"""

    def forward(self, **kwargs):
        return {
            "ctr_logits": None,
            "ctr_probs": None,
            "reward_pred": None
        }
    