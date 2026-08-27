from typing import Optional

from lm_eval.api.registry import register_model
from lm_eval.models.llada import LLaDA


@register_model("sdar")
class SDAR(LLaDA):
    """Generation-only lm-eval adapter for SDAR chat checkpoints."""

    MODEL_NAME = "sdar"

    def __init__(
        self,
        pretrained: str,
        block_length: int = 4,
        steps: int = 4,
        threshold: float = 0.85,
        mask_id: int = 151669,
        eos_id: Optional[int] = None,
        prefix_sparse: bool = False,
        sparse_dlm_selection_interval: int = 1,
        sparse_dlm_dense_fallback_mask_count: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            block_length=block_length,
            steps=steps,
            threshold=threshold,
            mask_id=mask_id,
            eos_id=eos_id,
            prefix_sparse=prefix_sparse,
            sparse_dlm_selection_interval=sparse_dlm_selection_interval,
            sparse_dlm_dense_fallback_mask_count=(
                sparse_dlm_dense_fallback_mask_count
            ),
            **kwargs,
        )
