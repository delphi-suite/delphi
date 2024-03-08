from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


@dataclass
class MambaArgs(MambaConfig):
    pass


class Mamba(MambaLMHeadModel):
    def __init__(self, params: MambaArgs) -> None:
        super().__init__(params)

    def forward(
        self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden_states)
        self.last_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-1
        )

        return logits

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """I don't want to implement this"""
        return 0
