__author__ = 'Shuo Yu'

from hmmlearn import hmm
from sklearn.cross_validation import KFold
import numpy as np

import util, func
import collections

# For some reason, the test performance on farseeing is not as good as the old hmm_correct script.
# Use that for test

def dict_inc(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


def sample_preprocessor(samples, func=None):
    """

    :param samples: a list of k lists, each list corresponding to the samples for a hmm model,
      each sample is a time series of data points, each data point has one or more vector components
    :param func: the function for converting the vectors, accepting n x 3 matrix or list
    :return:
    """
    if func is None:
        return samples
    return_list = []
    for i, sample_type_list in enumerate(samples): # the list of samples from the same type
        return_list.append([])
        for s in sample_type_list:
            return_list[i].append(func(s))
    return return_list


class HMMInstance:
    def __init__(self, N, transform, is_generalized, is_farseeing, pos_model_indices=None, pos_threshold=1, **kwargs):
        if is_generalized == True and not isinstance(pos_model_indices, collections.Iterable):
            raise Exception('Option is_generalized requires pos_model_indices')

        self.kwargs = kwargs
        self._N = N
        self._generalized = is_generalized
        self._pos_model_indices = pos_model_indices
        self._farseeing = is_farseeing
        self._transform = transform
        self._pos_threshold = pos_threshold

        self.hmm_models = []
        self.confusion_matrix = np.matrix([[0, 0], [0, 0]]) # Index 0 for ADL, 1 for Falls

    def load_hmm_model(self, file):
        import pickle
        with open(file, 'rb') as fh:
            self.hmm_models = pickle.load(fh)

    def dump_hmm_model(self, file):
        import pickle
        with open(file, 'wb') as fh:
            pickle.dump(self.hmm_models, protocol=3, file=fh)

    def hmm_classifier(self, sample):
        """
        TODO: modify this method to plot spec-sens curve

        :param sample: a matrix consisting of a single sample
        :return: an integer indicating the index of the hmm model with the highest probability
        """

        fall_score = -1e9
        non_score = -1e9
        fall_index = -1
        non_index = -1
        if not isinstance(sample, np.matrix):
            sample_mat = np.matrix(sample).T
        else:
            sample_mat = sample

        if self._generalized: # only two models in hmm_models
            for i, model in enumerate(self.hmm_models):
                score = model.score(sample_mat)
                if i == 1:
                    fall_score = score
                    fall_index = i
                else:
                    non_score = score
                    non_index = i
        else: # hopefully 12 models are in hmm_models, and self._pos_model_indices is set to [8, 9, 10, 11]
            for i, model in enumerate(self.hmm_models):
                score = model.score(sample_mat)
                if i in self._pos_model_indices:
                    if score > fall_score:
                        fall_score = score
                        fall_index = i
                else:
                    if score > non_score:
                        non_score = score
                        non_index = i

        # print(fall_score, non_score)
        # print(fall_index, non_index)

        # both fall_score and non_score < 0; thus, if the ratio < 1, fall_score > non_score
        if fall_score / non_score < self._pos_threshold:
            return fall_index
        else:
            return non_index

        # old method below

        # logprob = -1e9
        # index = -1
        # if not isinstance(sample, np.matrix):
        #     sample_mat = np.matrix(sample).T
        # else:
        #     sample_mat = sample
        #
        # for i in range(len(self.hmm_models)):
        #     model = self.hmm_models[i]
        #     score = model.score(sample_mat)
        #     if score > logprob:
        #         logprob = score
        #         index = i
        #
        # return index

    def preprocess(self, samples):
        if self._transform == 'g':
            return sample_preprocessor(samples, func.mat_to_g)
        elif self._transform == 'vc':
            return sample_preprocessor(samples, func.mat_to_vc)
        else:
            return samples

    def load_samples(self, training_samples, test_samples):
        if test_samples is None:
            _test_samples = training_samples
        else:
            _test_samples = test_samples

        self.training_samples = self.preprocess(training_samples)
        self.test_samples = self.preprocess(_test_samples)

    def train_hmm_model(self, training_samples=None):
        """
        :param samples: (preprocessed) a list of k lists, each list corresponding to the samples for a hmm model,
        each sample is a time series of data points, each data point has one or more vector components
        :return:
        """

        self.hmm_models = []

        if training_samples is not None:
            self.training_samples = self.preprocess(training_samples)

        # train hmm models
        if self._generalized == False:
            for samples_for_a_hmm in self.training_samples:
                lengths = [len(time_series) for time_series in samples_for_a_hmm]
                concaternated_samples = np.concatenate(samples_for_a_hmm)
                model = hmm.GaussianHMM(n_components=self._N)
                model.fit(concaternated_samples, lengths)
                self.hmm_models.append(model)
            # import pickle
            # with open('c:/hmm_models_spec.pkl', 'wb') as fh:
            #     pickle.dump(self.hmm_models, protocol=3, file=fh)
        else: # _generalized == True
            gen_lengths = [[], []]
            gen_concat_samples = [[], []]
            for sample_i in range(len(self.training_samples)):
                is_positive = 1 if sample_i in self._pos_model_indices else 0
                gen_lengths[is_positive].extend([len(time_series) for time_series in self.training_samples[sample_i]])
                concat_samples = np.concatenate(self.training_samples[sample_i])
                if gen_concat_samples[is_positive] == []:
                    gen_concat_samples[is_positive] = concat_samples
                else:
                    gen_concat_samples[is_positive] = np.concatenate((gen_concat_samples[is_positive], concat_samples))
            for i in range(2):
                model = hmm.GaussianHMM(n_components=self._N)
                model.fit(gen_concat_samples[i], gen_lengths[i])
                self.hmm_models.append(model)
            # import pickle
            # with open('c:/hmm_models_gen.pkl', 'wb') as fh:
            #     pickle.dump(self.hmm_models, protocol=3, file=fh)

    def evaluate(self, test_samples=None):
        self.true_index_list = []
        self.predicted_index_list = []
        self.true_label_list = []
        self.predicted_label_list = []

        self.confusion_matrix = np.matrix([[0, 0],
                                           [0, 0]])

        if test_samples is not None:
            self.test_samples = self.preprocess(test_samples)

        hmm_index = -1
        for samples_for_a_hmm in self.test_samples:
            hmm_index += 1
            for time_series in samples_for_a_hmm:
                eval_index = self.hmm_classifier(np.matrix(time_series))
                self.true_index_list.append(hmm_index)
                self.predicted_index_list.append(eval_index)

                if self._pos_model_indices is None:
                    self.true_label_list.append(hmm_index)
                    self.predicted_label_list.append(eval_index)
                else:
                    # self.true_label_list
                    if self._farseeing == True:
                        self.true_label_list.append(hmm_index)
                    else:
                        if hmm_index in self._pos_model_indices:
                            self.true_label_list.append(1)
                        else:
                            self.true_label_list.append(0)

                    # self.predicted_label_list
                    if self._generalized == False:
                        if eval_index in self._pos_model_indices:
                            self.predicted_label_list.append(1)
                        else:
                            self.predicted_label_list.append(0)
                    else: # self._generalized == True
                        self.predicted_label_list.append(eval_index)

        # print(self.true_label_list)
        # print(self.predicted_label_list)

        for true_label, predicted_label in zip(self.true_label_list, self.predicted_label_list):
            self.confusion_matrix[true_label, predicted_label] += 1

        # print(self.confusion_matrix)

    def get_prf(self):
        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]
        fn = self.confusion_matrix[1, 0]
        prec = tp / (tp + fp) if tp + fp != 0 else 0
        recl = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * prec * recl / (prec + recl) if prec + recl != 0 else 0
        return prec, recl, f1

    def get_pss(self):
        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]
        tn = self.confusion_matrix[0, 0]
        fn = self.confusion_matrix[1, 0]
        ppv = tp / (tp + fp) if tp + fp != 0 else 0
        sens = tp / (tp + fn) if tp + fn != 0 else 0
        spec = tn / (tn + fp) if tn + fp != 0 else 0
        return ppv, sens, spec


class HMMCrossValidator:
    def __init__(self, N=4, transform='vc', is_generalized=False, is_farseeing=False, pos_model_indices=(8, 9, 10, 11),
                 pos_threshold=1, **kwargs):
        if is_generalized == True and not isinstance(pos_model_indices, collections.Iterable):
            raise Exception('Option is_generalized requires pos_model_indices')

        self.kwargs = kwargs
        self._N = N
        self._generalized = is_generalized
        self._pos_model_indices = pos_model_indices
        self._pos_threshold = pos_threshold
        self._farseeing = is_farseeing
        self._transform = transform

        self.hmm_instances = []
        self.samples = []
        self.indices = []

    def gen_n_fold_indices(self, n_fold):
        """
        A wrapper function for sklearn.cross_validation.KFold, performing stratified n-fold split
        :param samples:
        :param n_fold:
        :return:
        """
        self.indices = []
        for i in range(len(self.samples)):
            # for each label (or type, model)
            self.indices.append([])
            for training_indices, test_indices in KFold(len(self.samples[i]), n_fold):
                # generate n-fold indices
                self.indices[i].append([training_indices, test_indices])

    def load_samples(self, samples):
        if self._transform == 'g':
            self.samples = sample_preprocessor(samples, func.mat_to_g)
        elif self._transform == 'vc':
            self.samples = sample_preprocessor(samples, func.mat_to_vc)
        else:
            self.samples = samples

    def execute(self, n_fold):
        self.gen_n_fold_indices(n_fold)
        for k in range(n_fold):
            training = []
            test = []
            for i in range(len(self.samples)):
                current_samples = np.array(self.samples[i])
                training.append(current_samples[self.indices[i][k][0]])
                test.append(current_samples[self.indices[i][k][1]])

            new_instance = HMMInstance(self._N, self._transform, self._generalized, self._farseeing,
                                       self._pos_model_indices, self._pos_threshold)
            new_instance.training_samples = training
            new_instance.test_samples = test
            new_instance.train_hmm_model()
            new_instance.evaluate()
            self.hmm_instances.append(new_instance)

    def report(self):
        total_confusion_matrix = np.matrix([[0, 0], [0, 0]])
        for each in self.hmm_instances:
            total_confusion_matrix = total_confusion_matrix + each.confusion_matrix
        print(total_confusion_matrix)


    def report_stats(self, fh=None, stats='pss'):
        verbose = False
        all = []
        n = len(self.hmm_instances)
        for each in self.hmm_instances:
            if stats == 'prf':
                stat = np.array(each.get_prf())
            elif stats == 'pss':
                stat = np.array(each.get_pss())
            all.append(stat)
        arr_all = np.array(all)
        avg = np.mean(arr_all, axis=0)
        stdev = np.std(arr_all, axis=0, ddof=1)
        output_str = ''
        if verbose:
            if stats == 'prf':
                output_str = '  Macro Average: Precision: {:.3f} ({:.3f}), Recall: {:.3f} ({:.3f}), F-measure: {:.3f} ({:.3f})'
            elif stats == 'pss':
                output_str = '  Macro Average: PPV: {:.3f} ({:.3f}), Sensitivity: {:.3f} ({:.3f}), Specificity: {:.3f} ({:.3f})'
        else:
            output_str = '{:.3f} ' * 6
        util.printf(output_str.format(avg[0], stdev[0], avg[1], stdev[1], avg[2], stdev[2]), fh)

    def report_prf(self, fh=None):
        self.report_stats(fh, stats='prf')

    def report_pss(self, fh=None):
        self.report_stats(fh, stats='pss')

    def load_samples_from_db(self, sensor_id=None):
        self.load_samples(util.load_samples_from_db())


class HMMManager:
    def __init__(self, N_range=None, transform=None, is_generalized=None, is_farseeing=False,
                 pos_model_indices=(8, 9, 10, 11), cross_validation=True, **kwargs):
        if is_generalized == True and not isinstance(pos_model_indices, collections.Iterable):
            raise Exception('Option is_generalized requires pos_model_indices')

        self.kwargs = kwargs
        self._N_range = N_range
        self._generalized = is_generalized
        self._pos_model_indices = pos_model_indices
        self._farseeing = is_farseeing
        self._transform = transform
        if cross_validation:
            self.cross_validation = True
            self.hmm_cv_instances = []
            self.samples = []
        else:
            self.cross_validation = False
            self.hmm_instance = []
            self.training = []
            self.test = []

    def load_samples_from_db(self, sensor_id=None):
        self.samples = util.load_samples_from_db()

    def execute_cv(self, n_fold, pos_threshold, fh=None):
        N_range_options = (4, 5) if self._N_range is None else self._N_range
        transform_options = (None, 'g', 'vc') if self._transform is None else self._transform
        generalized_options = (False, True) if self._generalized is None else self._generalized

        self.load_samples_from_db()

        for generalized in generalized_options:
            # print('Generalized = {}'.format(generalized))
            for transform in transform_options:
                # print('  Transform = {}'.format(transform))
                for N in range(N_range_options[0], N_range_options[1]):
                    # print('    N = {}'.format(N))
                    new_cv = HMMCrossValidator(N, transform, generalized, self._farseeing, self._pos_model_indices, pos_threshold)
                    new_cv.load_samples(self.samples)
                    new_cv.execute(n_fold=n_fold)
                    self.hmm_cv_instances.append(new_cv)

        self.report_pss(fh)

    def execute_cv_with_varying_pos_threshold(self, n_fold, pos_threshold_array, fh=None):
        for pos_threshold in pos_threshold_array:
            self.hmm_cv_instances = []
            self.execute_cv(n_fold, pos_threshold, fh)

    def train_test(self, pos_threshold=1, fh=None):
        N_range_options = (4, 5) if self._N_range is None else self._N_range
        # transform_options = (None, 'g', 'vc') if self._transform is None else self._transform
        transform_options = ('vc',) if self._transform is None else self._transform
        generalized_options = (False, True) if self._generalized is None else self._generalized

        self.training = util.load_samples_from_db()
        self.test = util.load_samples_from_farseeing()

        for generalized in generalized_options:
            # print('Generalized = {}'.format(generalized))
            for transform in transform_options:
                # print('  Transform = {}'.format(transform))
                for N in range(N_range_options[0], N_range_options[1]):
                    # print('    N = {}'.format(N))
                    hmm = HMMInstance(N, transform, generalized, self._farseeing, self._pos_model_indices, pos_threshold)
                    hmm.load_samples(self.training, self.test)
                    hmm.train_hmm_model()
                    hmm.evaluate()
                    self.hmm_instance.append(hmm)
        self.report_pss(fh)

    def load_model_test(self, pos_threshold=1, fh=None):
        self.test = util.load_samples_from_farseeing()

        hmm = HMMInstance(4, 'vc', is_generalized=False, is_farseeing=True, pos_model_indices=[8,9,10,11], pos_threshold=1)
        hmm.load_hmm_model('c:/hmm_models_spec.pkl')
        hmm.evaluate(self.test)
        print(hmm.get_pss())

    def report_prf(self, fh=None):
        if self.cross_validation:
            for each in self.hmm_cv_instances:
                util.printf('Generalized: {}, Transform: {}, N: {}, Threshold: {}'.format(each._generalized, each._transform,
                                                                                     each._N, each._pos_threshold), fh)
                each.report_prf(fh)
        else:
            for each in self.hmm_instance:
                util.printf('Generalized: {}, Transform: {}, N: {}, Threshold: {}'.format(each._generalized, each._transform,
                                                                                     each._N, each._pos_threshold), fh)
                output_str = '{:.3f} ' * 3
                util.printf(output_str.format(*each.get_prf()), fh)

    def report_pss(self, fh=None):
        if self.cross_validation:
            for each in self.hmm_cv_instances:
                util.printf('Generalized: {}, Transform: {}, N: {}, Threshold: {}'.format(each._generalized, each._transform,
                                                                                     each._N, each._pos_threshold), fh)
                each.report_pss(fh)
        else:
            for each in self.hmm_instance:
                util.printf('Generalized: {}, Transform: {}, N: {}, Threshold: {}'.format(each._generalized, each._transform,
                                                                                     each._N, each._pos_threshold), fh)
                output_str = '{:.3f} ' * 3
                util.printf(output_str.format(*each.get_pss()), fh)


if __name__ == '__main__':

    model = HMMManager()
    # model = HMMManager(is_farseeing=True, cross_validation=False)
    with open('c:/sample_output_10_fold_POOL_correct.txt', 'w') as fh:
        fh.flush()
        pta = (0.9, 0.95, 0.99, 0.995, 1, 1.005, 1.01, 1.05, 1.1)
        # pta = (1,)
        model.execute_cv_with_varying_pos_threshold(n_fold=10, pos_threshold_array=pta, fh=fh)
        # model.train_test(1, fh=fh)
        # model.load_model_test(1)
    print('Finished')
    # model = HMMCrossValidator()
    # model.load_samples_from_db()
    # model.execute(n_fold=5)
    # model.report_prf()
