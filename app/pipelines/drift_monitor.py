import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from datetime import datetime
import json

def detect_drift():
    reference = pd.read_csv("data/model_a_reference.csv")
    new_data = pd.read_csv("data/model_a.csv")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=new_data)
    summary = report.as_dict()

    drift_score = summary["metrics"][0]["result"]["drift_share"]
    drift_detected = drift_score > 0.1

    with open("drift_log.json", "a") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "drift_score": drift_score,
            "drift_detected": drift_detected
        }, f)
    return drift_detected

if __name__ == "__main__":
    if detect_drift():
        print("Drift detected, retraining needed")
    else:
        print("No significant drift")
