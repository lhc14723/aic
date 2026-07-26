"""Model components for AIC multi-modal detection."""

from aic_mm.models.fusion import TriModalStem, replace_first_layer_with_tri_modal_stem

__all__ = ["TriModalStem", "replace_first_layer_with_tri_modal_stem"]
