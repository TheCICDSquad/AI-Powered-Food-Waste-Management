from foodwaste.drift import detect_drift, simulate


def test_no_drift_on_normal_data(raw_data):
    result = detect_drift(raw_data, simulate("normal"))
    assert not result["drifted"].any(), result[result["drifted"]]


def test_heatwave_is_detected(raw_data):
    result = detect_drift(raw_data, simulate("heatwave"))
    drifted = set(result.loc[result["drifted"], "feature"])
    assert {"Weather", "Storage_Temperature"} <= drifted
