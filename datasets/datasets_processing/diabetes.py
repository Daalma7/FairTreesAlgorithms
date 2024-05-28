# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle


class DiabetesDataset(BaseDataset):
    def __init__(self, att,  data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'diabetes'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("original_data/diabetes2/preprocessed_diabetes.csv", sep=',', index_col=0)

        # df_base.loc[df_base['race'] == 'Caucasian','Race'] = 1
        df_base.loc[df_base['Race'] != 1, 'Race'] = 0
        # df_base.loc[df_base['race'] == 'AfricanAmerican','Race'] = 2
        # df_base.loc[df_base['race'] == 'Hispanic','Race'] = 3
        # df_base.loc[df_base['race'] == 'Asian','Race'] = 4
        # df_base.loc[df_base['race'] == 'Other','Race'] = 5
        # df_base.loc[df_base['gender'] == 'Male','Sex'] = 1
        # df_base.loc[df_base['gender'] == 'Female','Sex'] = 2
        df_base.loc[df_base['Sex'] == 2, 'Sex'] = 0


        # this is always binary
        df_base['binary_readmitted'] = df_base['Label']
        df_base.rename(columns={'Label': 'readmitted'}, inplace=True) # raw_df['readmitted'] == '>30'

        target_variable_ordinal = 'readmitted'
        target_variable_binary = 'binary_readmitted'


        self._ds = df_base

        self._explanatory_variables = ['Race', 'Sex', 'AgeGroup', 'A1CResult', 'Metformin', 'Chlorpropamide',
                                        'Glipizide', 'Rosiglitazone', 'Acarbose', 'Miglitol', 'DiabetesMed',
                                        'TimeInHospital', 'NumProcedures', 'NumMedications', 'NumEmergency']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male = 1
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
