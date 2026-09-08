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
        block_length: int = 32,
        steps: int = 32,
        threshold: float = 0.85,
        remasking_strategy: str = "sequential",
        eb_threshold: float = 0.35,
        mask_id: int = 151669,
        eos_id: Optional[int] = None,
        prefix_sparse: bool = False,
        sparse_dlm_selection_interval: int = 1,
        query_dense_threshold: int = 0,
        sparse_dlm_refresh_step: int = -1,
        sparse_dlm_selection_layer: int = 5,
        moe_expert_patch: bool = False,
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
            query_dense_threshold=(
                query_dense_threshold
            ),
            sparse_dlm_refresh_step=sparse_dlm_refresh_step,
            sparse_dlm_selection_layer=sparse_dlm_selection_layer,
            moe_expert_patch=moe_expert_patch,
            **kwargs,
        )
        self.remasking_strategy = str(remasking_strategy)
        self.eb_threshold = float(eb_threshold)

    def _extra_generation_kwargs(self) -> dict:
        kwargs = {"remasking_strategy": self.remasking_strategy}
        # The integrated LoSA/FOCUS drivers do not implement the sparse
        # runtime's entropy-bounded extension.  Keep that argument scoped to
        # the sparse adapter where it is meaningful.
        if self.runtime_mode == "sparse":
            kwargs["eb_threshold"] = self.eb_threshold
        return kwargs
