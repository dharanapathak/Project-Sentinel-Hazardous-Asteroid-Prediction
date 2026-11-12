import pandas as pd
from src.modeling import predict_one

class DummyModel:
    def predict(self, frame): return [1]
    def predict_proba(self, frame): return [[0.2, 0.8]]

def test_predict_one_returns_label_and_score():
    result = predict_one(DummyModel(), {"magnitude": 21.0}, ["magnitude"])
    assert result["prediction"] == "Hazardous"
    assert result["hazardous_class_probability_score"] == 0.8
