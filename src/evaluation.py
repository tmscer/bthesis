import numpy as np


def calculate_tp_fp_fn_tn(recommended, relevant, num_items):
    true_positives = len(set(recommended).intersection(relevant))
    false_positives = len(recommended) - true_positives

    assert true_positives + false_positives == len(recommended)

    false_negatives = len(relevant) - true_positives
    true_negatives = num_items - len(relevant) - false_positives

    assert true_negatives == num_items - len(recommended) - false_negatives

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "tn": true_negatives,
    }


def _test_calculate_tp_fp_fn_tn():
    tp_fp_fn_tn = calculate_tp_fp_fn_tn([1, 2, 3], [2, 3, 4], 5)

    assert tp_fp_fn_tn["tp"] == 2
    assert tp_fp_fn_tn["fp"] == 1
    assert tp_fp_fn_tn["fn"] == 1
    assert tp_fp_fn_tn["tn"] == 1


def calculate_precision_and_recall(tp_fp_fn_tn):
    tp = tp_fp_fn_tn["tp"]
    fp = tp_fp_fn_tn["fp"]
    fn = tp_fp_fn_tn["fn"]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def _test_precision_and_recall():
    # example from wikipedia
    tp_fp_fn_tn = {"tp": 5, "fp": 3, "fn": 7, "tn": 7}
    precision, recall = calculate_precision_and_recall(tp_fp_fn_tn)

    assert precision == 5 / 8
    assert recall == 5 / 12


_test_calculate_tp_fp_fn_tn()
_test_precision_and_recall()


def is_there_a_hit(recommended, relevant):
    return len(set(recommended).intersection(relevant)) > 0


def hitrate(hits):
    return np.sum(hits) / len(hits)


def _test_is_there_a_hit():
    assert is_there_a_hit([1, 2, 3], [3, 4, 5])
    assert not is_there_a_hit([1, 2, 3], [4, 5, 6])


def _test_hitrate():
    assert hitrate([True, True, False, False, False]) == 0.4
    assert hitrate([1.0, 1.0, 0.0, 0.0, 0.0]) == 0.4

    assert hitrate([False, False, False]) == 0.0
    assert hitrate([0.0, 0.0, 0.0]) == 0.0

    assert hitrate([True, True]) == 1.0
    assert hitrate([1.0, 1.0]) == 1.0


_test_is_there_a_hit()
_test_hitrate()
