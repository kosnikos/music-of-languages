import numpy as np
import pytest

from musiclang.features.base import FeatureExtractor, ConstantExtractor


def test_constant_extractor_implements_interface():
    ex = ConstantExtractor(value=1.5)
    assert isinstance(ex, FeatureExtractor)
    assert ex.name == "constant"
    out = ex.extract(np.zeros(100, dtype=np.float32), sr=16_000)
    assert out == {"constant": 1.5}


def test_feature_extractor_is_abstract():
    with pytest.raises(TypeError):
        FeatureExtractor()  # cannot instantiate ABC
