import pytest
from storypoint.synthetic import generate,FEATURES
from storypoint.model import train,predict

def test_reproducibility():
    assert generate(160,seed=3).equals(generate(160,seed=3))
def test_conformal_train():
    m=train(300,seed=4)
    assert m.metrics["train_rows"]+m.metrics["calibration_rows"]+m.metrics["test_rows"]==300
    assert 0<=m.metrics["empirical_coverage"]<=1
    assert m.cutoff>=0
def test_prediction_bounds():
    m=train(240)
    x=predict(m,{f:2 for f in FEATURES})
    assert 1<=x["lower"]<=x["estimate"]<=x["upper"]<=13
    with pytest.raises(ValueError):predict(m,{"dependencies":1})
def test_invalid_n_and_alpha():
    with pytest.raises(ValueError):generate(50)
    with pytest.raises(ValueError):train(120,alpha=1)
