import pandas as pd
import pytest
from src.preprocessing import prepare_xy, parse_target

def test_prepare_xy_excludes_identifiers_and_target():
    df = pd.DataFrame({
        "id":[1,2,3,4], "name":["a","b","c","d"],
        "absolute_magnitude":[20.,21.,22.,23.],
        "hazardous":[False,True,False,True]
    })
    X, y = prepare_xy(df)
    assert list(X.columns) == ["absolute_magnitude"]
    assert y.tolist() == [0,1,0,1]

def test_target_rejects_unknown_labels():
    with pytest.raises(ValueError):
        parse_target(pd.Series(["maybe", "no"]))

def test_missing_target_fails():
    with pytest.raises(ValueError):
        prepare_xy(pd.DataFrame({"x":[1,2]}))
