import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from datasets_processing._dataset import BaseDataset, get_the_middle


class ObesityDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'ordinal'
        self._name = 'obesity'
        self._att = att
        self.preprocess()

    def preprocess(self):
        df = pd.read_csv("original_data/obesity/ObesityDataSet_raw_and_data_sinthetic.csv", sep=',')

        # We categorized as 5 the obesity types 2 and 3.
        label_map = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 5}

        cat = 'categorical'
        num = 'numeric'
        unique_counter = df.apply(lambda col: len(set(col)), axis=0)
        variables = {}
        for key, val in unique_counter.items():
            variables[key] = (cat, val) if val < 20 else (num, None)

        target = np.array([label_map[t] for t in df['NObeyesdad']])
        gender = (df['Gender'] != 'Female').astype(float)

        df_ = df.drop(['NObeyesdad', 'Gender'], axis=1)
        df_.loc[df_['CALC'] == 'Always', 'CALC'] = 'Frequently'

        # categorical columns
        catcols = [var for var in variables if variables[var][0] == cat if var in df_]
        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        df_cat = cat_encoder.fit_transform(df_[catcols])
        cat_new_cols = np.concatenate(
            [[cat] if len(item) == 2 else [cat + '_' + cn for cn in item[1:]] for cat, item in
             zip(catcols, cat_encoder.categories_)]).tolist()
        cat_df = pd.DataFrame(df_cat, columns=cat_new_cols)

        # numerical columns
        numcols = [var for var in variables if variables[var][0] == num if var in df_]

        dataset = pd.concat([df_[numcols], cat_df], axis=1)

        dataset['Gender'] = gender
        dataset['NObeyesdad'] = target

        # computing the binary outcome
        dataset['NObeyesdad_binary'] = dataset['NObeyesdad'] >= 4
        dataset['NObeyesdad_binary'] = dataset['NObeyesdad_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)


        self._ds = dataset

        self._explanatory_variables = ['family_history_with_overweight', 'FAVC', 'CAEC_Frequently', 'CAEC_Sometimes',
                                       'CAEC_no', 'SMOKE', 'SCC', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
                                       'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking',
                                       'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Gender']

        if self.outcome_type == 'binary':
            self._outcome_label = 'NObeyesdad_binary'  # [0, 1]
        else:
            self._outcome_label = 'NObeyesdad'  # [0-5]

        self._favorable_label_binary = [0]
        self._favorable_label_continuous = [0, 1, 2, 3]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'NObeyesdad'
        self._binary_label_name = 'NObeyesdad_binary'

        self._cut_point = 4  # is >= all above is 1
        self._non_favorable_label_continuous = [4, 5]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)

        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)

