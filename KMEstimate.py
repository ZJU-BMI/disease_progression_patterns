import numpy as np


def get_km_scores(times, labels, fail_code=1, max_times=100):
    """
    # estimate KM survival rate
    :param times: ndarray, shape(num_subject, ), event times or censoring times, shape
    :param labels: ndarray, shape(num_subject, ), event labels
    :param fail_code: event_id
    :param max_times: max observation time
    :return:
    """
    N = len(times)
    # Sorting T and E in ascending order by T
    order = np.argsort(times)
    T = times[order]
    E = labels[order]
    max_T = int(np.max(T)) + 1
    max_T = max(max_times, max_T)
    # calculate KM survival rate at time 0-T_max
    km_scores = np.ones(max_T)
    n_fail = 0
    n_rep = 0

    for i in range(N):

        if E[i] == fail_code:
            n_fail += 1

        if i < N - 1 and T[i] == T[i + 1]:
            n_rep += 1
            continue

        km_scores[int(T[i])] = 1. - n_fail / (N - i + n_rep)
        n_fail = 0
        n_rep = 0

    for i in range(1, max_T):
        km_scores[i] = km_scores[i - 1] * km_scores[i]

    return km_scores


def pick_surv(score, step):
    if step > len(score):
        return score

    times = []
    scores = []
    num = int((len(score) + step - 1) / step)
    for i in range(num):
        times.append(i * step)
        scores.append(score[i * step])

    return times, scores


def brier_score_surv(pred, time, label, eval_time):
    N = 0
    bs = 0
    for p, t, y in zip(pred, time, label):
        if t < eval_time and y == 0:
            continue
        N += 1
        if t <= eval_time and y == 1:
            bs += (1 - p)**2
        else:
            bs += p**2
    if N > 0:
        return 1. * bs / N
    else:
        return None
