"""Model components for AIC multi-modal detection."""

from aic_mm.models.fusion import (
    TriModalStem,
    TriModalStemV2,
    fusion_stem_type,
    initialize_stem_from_rgb_conv,
    replace_first_layer_with_tri_modal_stem,
)

__all__ = [
    "TriModalStem",
    "TriModalStemV2",
    "fusion_stem_type",
    "initialize_stem_from_rgb_conv",
    "replace_first_layer_with_tri_modal_stem",
]
