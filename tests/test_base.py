from base import MyHMM
import pytest


def test_init():
    with pytest.raises(TypeError):
        obj = MyHMM()
    
    class MyHMMTest(MyHMM):
        HMM = None
        HMM_config = None
        def set_fitted_params(cls, config: dict):
            pass
        def get_fitted_params(self) -> dict:
            pass
    obj = MyHMMTest()
    assert hasattr(obj, 'timestamp')
    obj = MyHMMTest(timestamp=1)
    assert obj.timestamp_ == 1


