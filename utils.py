import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, auc
from scipy import stats

# standard or normalize features
def get_normalization(X, norm_mode='standard'):
    num_Patient, num_Feature = np.shape(X)
    if norm_mode is None:
        return X
    if norm_mode == 'standard':  # zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            else:
                X[:, j] = (X[:, j] - np.mean(X[:, j]))
    elif norm_mode == 'normal':  # min-max normalization
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
    else:
        print("INPUT MODE ERROR!")

    return X


def get_km_scores(times, labels, fail_code, sort=False):
    """
    # estimate KM survival rate
    :param times: ndarray, shape(num_subject, ), event times or censoring times, shape,
    :param labels: ndarray, shape(num_subject, ), event labels
    :param fail_code: event_id
    :param sort: whether sort by times, default False (we assume that the time is sorted in ascending order)
    :return:
    """
    N = len(times)
    times = np.reshape(times, [-1])
    labels = np.reshape(labels, [-1])
    # Sorting T and E in ascending order by T
    if sort:
        order = np.argsort(times)
        T = times[order]
        E = labels[order]
    else:
        T = times
        E = labels
    max_T = int(np.max(T)) + 1

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


def get_bh_mask(time, label, num_times):
    # Tensor to numpy
    time = time.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    N = len(time)
    T = np.reshape(time, [N])
    E = np.reshape(label, [N])
    mask = np.zeros([N, num_times])

    for i in range(N):
        if E[i] > 0:
            mask[i, int(T[i])] = 1
        else:
            mask[i, int(T[i] + 1):] = 1
    return mask


def cluster_acc(pred, label):
    D = max(pred.max(), label.max()) + 1
    w = np.zeros([D, D], dtype=np.int32)
    for i in range(pred.size):
        w[pred[i], label[i]] += 1
    row_ind, col_ind = linear_sum_assignment(np.max(w) - w)

    return w[row_ind, col_ind].sum() * 1. / pred.size, w


# this saves the current hyperparameters
def save_params(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            f.write('%s:%s\n' % (key, value))


def load_params(filename):
    params = dict()
    with open(filename) as f:
        def is_float(x):
            try:
                num = float(x)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key, value = line.strip().split(':', 1)
                if value.isdigit():
                    params[key] = int(value)
                elif is_float(value):
                    params[key] = float(value)
                elif value == 'None':
                    params[key] = None
                else:
                    params[key] = value
            else:
                pass  # deal with bad lines of text here
    return params


def survival_plot(time, label, clusters, save_path=None, show=True, nclusters=2):

    for k in range(nclusters):
        idx1 = np.where(clusters == k)[0]
        print(len(idx1))
        T1 = time[idx1]
        E1 = label[idx1]
        kmf = KaplanMeierFitter()
        kmf.fit(T1, E1)
        if k == 0:
            ax = kmf.plot_survival_function(label='cluster0')
        else:
            ax = kmf.plot_survival_function(ax=ax, label='cluster0')

    # idx2 = np.where(clusters == 1)[0]
    # print(len(idx2))
    # T2 = time[idx2]
    # E2 = label[idx2]
    # kmf.fit(T2, E2)
    ax = kmf.plot_survival_function(ax=ax, label='cluster1')

    # stat = logrank_test(durations_A=T1, durations_B=T2, event_observed_A=E1, event_observed_B=E2)
    # print(stat.p_value)
    plt.title('Survival curve')
    if save_path is not None:
        plt.savefig(save_path + '/surv_curve.png')
    if show:
        plt.show()


def plot_ROCs(fileList, names, colors=None, save=True, save_path=None, show=False,
              dpi=600, data_name=None, method=None):
    plt.figure(figsize=(6, 6), dpi=dpi)

    for (filename, name, color) in zip(fileList, names, colors):
        df = pd.read_csv(filename)
        label = np.asarray(df['label'])
        pred = np.asarray(df['prediction'])
        fpr, tpr, thres = roc_curve(label, pred, pos_label=1)

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=color)
        plt.plot([0, 1], [0, 1], '--', lw=3, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tick_params(labelsize=10)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        if method is None:
            plt.title('ROC Curves ({})'.format('CN→MCI' if data_name == 'cntomci' else 'MCI→AD'), fontsize=14)
        else:
            plt.title('ROC Curves with {} ({})'.format(method, 'CN→MCI' if data_name == 'cntomci' else 'MCI→AD'),
                      fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout(h_pad=5, w_pad=4)

    if save:
        if save_path is None:
            plt.savefig('roc.jpg', dpi=dpi)
        else:
            if data_name is None:
                plt.savefig(save_path + '/roc_curve.jpg', dpi=dpi)
            else:
                plt.savefig(save_path + '/roc_curve_{}.jpg'.format(data_name), dpi=dpi)
    if show:
        plt.show()
    return plt


def get_num_eval(time, label, eval_time):
    num_eval = 0
    N = len(time)
    for i in range(N):
        # positive(died) at eval_time
        if time[i] <= eval_time and label[i] == 1:
            num_eval += 1
        # non-positive(survival) at eval_time
        elif time[i] > eval_time:
            num_eval += 1

    return num_eval


def get_tf_positive(time, label, threshold, pred_prob, eval_time):
    tp = fp = 0
    num_eval = 0
    N = len(pred_prob)
    for i in range(N):
        # positive(died) at eval_time
        if time[i] <= eval_time and label[i] == 1:
            num_eval += 1
            if pred_prob[i] >= threshold:
                tp += 1
        # non-positive(survival) at eval_time
        elif time[i] > eval_time:
            num_eval += 1
            if pred_prob[i] >= threshold:
                fp += 1
    return tp, fp, num_eval


def cal_net_benefit(time, label, threshold, pred_prob, eval_time):
    if threshold == 0:
        threshold = 1e-8
    elif threshold == 1:
        threshold = 1. - 1e-8
    time = np.reshape(time, [-1])
    label = np.reshape(label, [-1])
    tp, fp, num_eval = get_tf_positive(time, label, threshold, pred_prob, eval_time=eval_time)
    if num_eval == 0:
        raise ValueError('Can not calculate net benefit')
    theta = threshold / (1 - threshold)
    res = tp * 1. / num_eval - fp * 1. / num_eval * theta
    return res


def mean_interval(mean=None, std=None, sig=None, n=None, confidence=0.95):
    alpha = 1 - confidence
    z_score = stats.norm.isf(alpha / 2)
    t_score = stats.t.isf(alpha / 2, df=(n - 1))
    lower_limit = mean
    upper_limit = mean

    if n >= 30 and sig is not None:
        me = z_score * sig / np.sqrt(n)
        lower_limit = mean - me
        upper_limit = mean + me

    if n >= 30 and sig is None:
        me = z_score * std / np.sqrt(n)
        lower_limit = mean - me
        upper_limit = mean + me

    if n < 30 and sig is None:
        me = t_score * std / np.sqrt(n)
        lower_limit = mean - me
        upper_limit = mean + me

    return lower_limit, upper_limit
