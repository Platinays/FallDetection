__author__ = 'Shuo Yu'

import util, func
import numpy as np


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


def pool():
    x_list = []
    y_list = []

    pool_samples = util.load_samples_from_db()

    for i, samples in enumerate(pool_samples):
        for each in samples:
            if len(each) > 0:
                x_list.append(func.identify_peak(np.array(each))[0])
                y_list.append(0 if i < 8 else 1)

    print(x_list)
    print(y_list)

    thresholds = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)

    with open('c:/sample_output_thres.txt', 'w') as fh:
        for threshold in thresholds:
            util.printf('Threshold: {}'.format(threshold), fh)
            res = []
            for x in x_list:
                if x >= threshold:
                    res.append(1)
                else:
                    res.append(0)

            util.printf(str(classification_summary(y_list, res)), fh)


def farseeing():
    test_x_list = []
    test_y_list = []

    far_samples = util.load_samples_from_farseeing()

    for i, samples in enumerate(far_samples):
        for each in samples:
            if len(each) > 0:
                test_x_list.append(func.identify_peak(np.array(each))[0])
                test_y_list.append(i)

    thresholds = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)

    with open('c:/sample_output_thres_far.txt', 'w') as fh:
        for threshold in thresholds:
            util.printf('Threshold: {}'.format(threshold), fh)
            res = []
            for x in test_x_list:
                if x >= threshold:
                    res.append(1)
                else:
                    res.append(0)

            util.printf(str(classification_summary(test_y_list, res)), fh)

if __name__ == '__main__':
    farseeing()