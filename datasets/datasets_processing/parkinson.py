import pandas as pd

# from aif360 import datasets as ds
from datasets_processing._dataset import BaseDataset


class ParkinsonDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'continuous'
        self._name = 'parkinson'
        self._att = att
        self.preprocess()

    def preprocess(self):
        df = pd.read_csv("original_data/parkinson/parkinsons_updrs.data", sep=',')

        dataset = df.drop(['subject#', 'motor_UPDRS', 'test_time'], axis=1)

        total_UPDRS = dataset['total_UPDRS']
        dataset.drop(['total_UPDRS'], axis=1, inplace=True)
        dataset['total_UPDRS'] = total_UPDRS

        # computing the binary outcome
        dataset['total_UPDRS_binary'] = dataset['total_UPDRS'] > 17.1
        dataset['total_UPDRS_binary'] = dataset['total_UPDRS_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['age', 'sex', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP',
                                       'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3',
                                       'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
                                       'DFA', 'PPE']

        if self.outcome_type == 'binary':
            self._outcome_label = 'total_UPDRS_binary'  # [0, 1]
        else:
            self._outcome_label = 'total_UPDRS'

        # https://jamanetwork.com/journals/jamaneurology/fullarticle/799064#:~:text=Estimates%20for%20the%20UPDRS%20total,and%20response%20to%20therapeutic%20interventions.
        self._favorable_label_binary = [0]
        self._favorable_label_continuous = lambda x: x < 17.1

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'total_UPDRS'
        self._binary_label_name = 'total_UPDRS_binary'

        self._cut_point = 17.1  # is >= all above is 1
        self._non_favorable_label_continuous = lambda x: x >= 17.1

        self._fav_dict = None
        self._nonf_dict = None

        self._middle_fav = None
        self._middle_non_fav = None
