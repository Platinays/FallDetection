__author__ = 'Shuo Yu'

import util, func
import numpy as np
import math

def angle_measure(arr):
    ret_list = []
    g_vec = func.calc_gravity_o(arr)
    for vec in arr:
        dot_product = np.dot(vec, g_vec) / np.linalg.norm(vec) / np.linalg.norm(g_vec)
        if dot_product >= 1:
            ret_list.append(0)
        else:
            ret_list.append(math.acos(dot_product) * 180 / math.pi)

    return ret_list


def max_angle_time(angle_list, UPV_index):
    window = 37

    max_angle = np.amax(np.array(angle_list[
                                 max(UPV_index - window, 0): UPV_index + window
                                 ]))
    max_angle_index = np.argmax(np.array(angle_list[
                                         max(UPV_index - window, 0) : UPV_index + window
                                         ])) + UPV_index - window

    return max_angle, max_angle_index - UPV_index


def generate_features(ret_list):
    if not isinstance(ret_list, np.ndarray):
        arr = np.array(ret_list)
    else:
        arr = ret_list
    # mag_list = func.calc_magnitude(arr)
    angle_list = angle_measure(arr)

    UPV, UPV_index = func.identify_peak(arr)
    LPV, LPV_index = func.identify_valley_before_peak(arr)
    max_angle, max_angle_t = max_angle_time(angle_list, UPV_index)

    return UPV, LPV, max_angle, max_angle_t


def return_index_by_threshold(arr, threshold):
    ret_list = []
    for each in arr:
        if each[1] >= threshold:
            ret_list.append(1)
        else:
            ret_list.append(0)
    return ret_list


def classification_summary(true_y_list, predicted_y_list):
    confusion_matrix = np.matrix([[0, 0],
                                  [0, 0]])
    for true_y, predicted_y in zip(true_y_list, predicted_y_list):
        confusion_matrix[true_y, predicted_y] += 1

    print(confusion_matrix)

    tp = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    tn = confusion_matrix[0, 0]
    fn = confusion_matrix[1, 0]
    ppv = tp / (tp + fp) if tp + fp != 0 else 0
    sens = tp / (tp + fn) if tp + fn != 0 else 0
    spec = tn / (tn + fp) if tn + fp != 0 else 0
    return ppv, sens, spec


def cross_validation(n_fold, x_list, y_list, thresholds=(0.5, ), fh=None):
    from sklearn.cross_validation import StratifiedKFold

    kf = StratifiedKFold(y_list, n_fold)

    if isinstance(x_list, np.ndarray):
        x = x_list
    else:
        x = np.array(x_list)

    if isinstance(y_list, np.ndarray):
        y = y_list
    else:
        y = np.array(y_list)

    models = []

    for train_index, test_index in kf:
        # from sklearn.tree import DecisionTreeClassifier
        # clf = DecisionTreeClassifier(min_samples_leaf=5)
        # from sklearn.svm import SVC
        # clf = SVC(kernel='linear', probability=True)
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(x[train_index], y[train_index])
        models.append(clf)

    for threshold in thresholds:
        util.printf('Threshold: {}'.format(threshold), fh)
        results = []
        for i, (train_index, test_index) in enumerate(kf):
            predicted_y_proba = models[i].predict_proba(x[test_index])
            predicted_y_list = return_index_by_threshold(predicted_y_proba.tolist(), threshold=threshold)
            results.append(classification_summary(y[test_index].tolist(), predicted_y_list))

        results_array = np.array(results)
        util.printf(str(np.mean(results_array, axis=0)), fh)
        util.printf(str(np.std(results_array, axis=0, ddof=1)), fh)


def execute_cv():
    x_list = []
    y_list = []

    pool_samples = util.load_samples_from_db()

    for i, samples in enumerate(pool_samples):
        for each in samples:
            if len(each) > 0:
                x_list.append(generate_features(each))
                y_list.append(0 if i < 8 else 1)

    with open('c:/sample_output_10_fold_nb.txt', 'w') as fh:
        fh.flush()
        pta = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        cross_validation(10, x_list, y_list, pta, fh)


def train_test(train_x_list, train_y_list, test_x_list, test_y_list, thresholds=(0.5,), fh=None):
    train_x = train_x_list if isinstance(train_x_list, np.ndarray) else np.array(train_x_list)
    train_y = train_y_list if isinstance(train_y_list, np.ndarray) else np.array(train_y_list)
    test_x = test_x_list if isinstance(test_x_list, np.ndarray) else np.array(test_x_list)
    test_y = test_y_list if isinstance(test_y_list, np.ndarray) else np.array(test_y_list)
    # from sklearn.tree import DecisionTreeClassifier
    # clf = DecisionTreeClassifier(min_samples_leaf=5)
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', probability=True)
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    clf.fit(train_x, train_y)

    for threshold in thresholds:
        util.printf('Threshold: {}'.format(threshold), fh)

        predicted_y_proba = clf.predict_proba(test_x)
        predicted_y_list = return_index_by_threshold(predicted_y_proba.tolist(), threshold=threshold)
        util.printf(("{:.3f} " * 3).format(*classification_summary(test_y.tolist(), predicted_y_list)), fh)


def execute_train_test():
    x_list = []
    y_list = []

    pool_samples = util.load_samples_from_db()

    for i, samples in enumerate(pool_samples):
        for each in samples:
            if len(each) > 0:
                x_list.append(generate_features(each))
                y_list.append(0 if i < 8 else 1)

    test_x_list = []
    test_y_list = []

    far_samples = util.load_samples_from_farseeing()

    for i, samples in enumerate(far_samples):
        for each in samples:
            if len(each) > 0:
                test_x_list.append(generate_features(each))
                test_y_list.append(i)

    with open('c:/sample_output_10_fold_svm_far.txt', 'w') as fh:
        fh.flush()
        pta = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        train_test(x_list, y_list, test_x_list, test_y_list, pta, fh)

if __name__ == '__main__':
    execute_train_test()
