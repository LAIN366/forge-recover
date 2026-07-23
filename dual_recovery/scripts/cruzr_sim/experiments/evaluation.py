"""Diagnosis accuracy, binary detection scores, and confusion matrices."""

from collections import Counter, defaultdict


EXPECTED_DIAGNOSIS = {
    "none": "none",
    "slip": "grasp_slip",
    "missed_grasp": "missed_grasp",
    "target_shift": "target_displacement",
    "contact_noise": "none",
    "collision_event": "collision",
    "ik_failure": "ik_failure",
    "planning_failure": "planning_failure",
    "sensor_dropout": "sensor_fault",
}


def final_prediction(episode):
    """Use post-probe confirmation when available, otherwise raw diagnosis."""
    if episode.confirmed_fault != "none":
        return episode.confirmed_fault
    if episode.policy == "diagnosis_only":
        return episode.first_diagnosis
    return "none"


def evaluate_diagnosis(episodes):
    by_policy = defaultdict(list)
    for episode in episodes:
        by_policy[episode.policy].append(episode)
    summaries = []
    confusion_rows = []
    for policy, items in sorted(by_policy.items()):
        counts = Counter()
        correct = 0
        true_positive = false_positive = false_negative = 0
        for item in items:
            expected = EXPECTED_DIAGNOSIS.get(item.fault, item.fault)
            predicted = final_prediction(item)
            counts[(expected, predicted)] += 1
            correct += expected == predicted
            expected_anomaly = expected != "none"
            predicted_anomaly = predicted != "none"
            true_positive += expected_anomaly and predicted_anomaly
            false_positive += not expected_anomaly and predicted_anomaly
            false_negative += expected_anomaly and not predicted_anomaly
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summaries.append({
            "policy": policy,
            "episodes": len(items),
            "classification_accuracy": correct / len(items) if items else 0.0,
            "anomaly_precision": precision,
            "anomaly_recall": recall,
            "anomaly_f1": f1,
        })
        for (expected, predicted), count in sorted(counts.items()):
            confusion_rows.append({
                "policy": policy,
                "expected": expected,
                "predicted": predicted,
                "count": count,
            })
    return summaries, confusion_rows
