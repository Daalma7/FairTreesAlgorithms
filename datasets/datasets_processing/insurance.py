import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from datasets_processing._dataset import BaseDataset


class InsuranceDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'continuous'
        self._name = 'insurance'
        self._att = att
        self.preprocess()

    def preprocess(self):
        df = pd.read_csv("original_data/insurance/insurance.csv", sep=',')

        cat_vars = ['sex', 'smoker', 'region']
        num_vars = ['age', 'children', 'bmi', 'charges']

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        cat_data = cat_encoder.fit_transform(df[cat_vars])

        catnewcols = np.concatenate(
            [[cat] if len(item) == 2 else [cat + '_' + cn for cn in item[1:]] for cat, item in
             zip(cat_vars, cat_encoder.categories_)]).tolist()

        cat_df = pd.DataFrame(cat_data, columns=catnewcols)

        dataset = pd.concat([df[num_vars], cat_df], axis=1)

        # computing the binary outcome - 3rd percentile
        dataset['charges_binary'] = dataset['charges'] > 40000
        dataset['charges_binary'] = dataset['charges_binary'].astype(int)
        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['age', 'children', 'bmi', 'sex', 'smoker',
                                       'region_northwest', 'region_southeast', 'region_southwest']

        if self.outcome_type == 'binary':
            self._outcome_label = 'charges_binary'  # [0, 1]
        else:
            self._outcome_label = 'charges'

        self._favorable_label_binary = [0]
        self._favorable_label_continuous = lambda x: x <= 40000

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'charges'
        self._binary_label_name = 'charges_binary'

        self._cut_point = 40001  # is >= all above is 1
        self._non_favorable_label_continuous = lambda x: x > 40000

        self._fav_dict = None
        self._nonf_dict = None

        self._middle_fav = None
        self._middle_non_fav = None

