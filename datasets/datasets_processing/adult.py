# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle


class AdultDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'adult'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("original_data/adult/preprocessed_adult.csv", sep=',', index_col=0)

        label = df_base['label']

        df_base.drop(['label'], axis=1, inplace=True)
        df_base['income'] = label

        # this is always binary
        df_base['binary_income'] = df_base['income']

        target_variable_ordinal = 'income'
        target_variable_binary = 'binary_income'


        # protected att
        # processed_df.loc[df['sex'] == 'Male', 'Sex'] = 1
        df_base.loc[df_base['Sex'] == 2, 'Sex'] = 0
        # processed_df.loc[df['race'] == 'White', 'Race'] = 1
        # processed_df.loc[df['race'] == 'Non-white', 'Race'] = 2
        df_base.loc[df_base['Race'] == 2, 'Race'] = 0
        # processed_df.loc[df['native_country'] == 'United-States', 'NativeCountry'] = 1
        # processed_df.loc[df['native_country'] == 'Non-United-Stated', 'NativeCountry'] = 2
        df_base.loc[df_base['NativeCountry'] == 2, 'NativeCountry'] = 0


        self._ds = df_base

        self._explanatory_variables = [ 'Sex', 'Race', 'AgeGroup', 'NativeCountry', 'WorkClass',
                                        'EducationNumber', 'EducationLevel', 'MaritalStatus', 'Occupation',
                                        'Relationship', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # TODO change this to correspond with the paper values
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
