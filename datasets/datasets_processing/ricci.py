# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle



column_mapping = {'Position': {'Captain': 1, 'Lieutenant': 0}, 'Race': {'W': 1, 'B':0, 'H':0}}

class RicciDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'ricci'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("original_data/ricci/ricci.csv", header=0)

        # Convert columns to numeric, if possible
        for col, mapping in column_mapping.items():
            df_base[col] = df_base[col].map(mapping)


        conversion_col = df_base['Combine'].astype(float).astype(int)
        y = np.where(conversion_col >= 70, 0, 1)
        df_base['Combine'] = y

        # this is always binary
        df_base['binary_Combine'] = df_base['Combine']

        target_variable_ordinal = 'Combine'
        target_variable_binary = 'binary_Combine'

        self._ds = df_base

        self._explanatory_variables = ['Oral', 'Written', 'Race', 'Position']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # caucasian
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = target_variable_ordinal
        self._binary_label_name = target_variable_binary

        self._cut_point = 1
        self._non_favorable_label_continuous = [0]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)
        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)

