from base import MyHMM
import pytest


def test_init():
    with pytest.raises(TypeError):
        obj = MyHMM()
    
    class MyHMMTest(MyHMM):
        HMM = None
        HMM_config = None
        def from_config(cls, config: dict):
            pass
        def get_config(self) -> dict:
            pass
    obj = MyHMMTest()
    assert hasattr(obj, 'timestamp')
    obj = MyHMMTest(timestamp=1)
    assert obj.timestamp == 1


