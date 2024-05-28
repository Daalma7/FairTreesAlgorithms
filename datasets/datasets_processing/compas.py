# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from datasets_processing._dataset import BaseDataset, get_the_middle


class CompasDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0 , outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'ordinal'
        self._name = 'compas'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df = pd.read_csv("original_data/compas/compas-scores-two-years.csv", sep=',')
        columns = ['sex', 'age', 'age_cat', 'race',
                 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                 'priors_count', 'c_charge_degree', 'c_charge_desc',
                 'two_year_recid', 'is_recid', 'days_b_screening_arrest',
                 'decile_score', 'score_text']
        filtered_df = df[columns].loc[(df['days_b_screening_arrest'] <= 30)
                                      & (df['days_b_screening_arrest'] >= -30)
                                      & (df['is_recid'] != -1)
                                      & (df['c_charge_degree'] != "O")
                                      & (df['score_text'].notnull())]

        filtered_df.reset_index(inplace=True, drop=True)

        # sex
        gender = filtered_df['sex'].apply(lambda x: 1 if x == 'Female' else 0)
        df_gender = gender.to_frame()

        # misdemeanor
        charge = filtered_df['c_charge_degree'].apply(lambda x: 1 if x == 'M' else 0)
        df_charge = charge.to_frame()
        df_charge.rename(columns={'c_charge_degree': 'misdemeanor'}, inplace=True)

        # race
        race = filtered_df['race'].apply(lambda x: 1 if x == 'Caucasian' else (0 if x == 'African-American' else 2))
        # race = filtered_df['race'].apply(lambda x: 0 if x == 'African-American' else 1)
        df_race = race.to_frame()

        # Binarizing race and age_cat
        lb_style = LabelBinarizer()
        lb_age = lb_style.fit_transform(filtered_df['age_cat'])
        df_age = pd.DataFrame(lb_age, columns=lb_style.classes_)

        # For the binary score we are using Low versus Medium and High
        df_base = filtered_df[['two_year_recid', 'priors_count', 'score_text']]
        df_decile = filtered_df[['decile_score']]
        df_base = df_base.join(df_age).join(df_race).join(df_gender).join(df_charge).join(df_decile)
        # High >= 8
        df_base['binary_score'] = df_base['score_text'].apply(lambda x: 1 if x == 'High' or x == 'Medium' else 0)
        df_base.drop(['score_text'], axis=1, inplace=True)

        # We will take into consideration only Caucasian and African american values of the race attribute.
        df_base = df_base.drop(df_base[df_base.race == 2].index)
        df_base.reset_index(inplace=True, drop=True)

        target_variable_ordinal = 'decile_score'
        target_variable_binary = 'binary_score'

        self._ds = df_base

        self._explanatory_variables = ['two_year_recid', 'priors_count', '25 - 45', 'Greater than 45',
                                      'Less than 25', 'race', 'sex', 'misdemeanor']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [0]
        self._favorable_label_continuous = [1, 2, 3, 4]

        # TODO change this to correspond with the paper values
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # caucasian
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = target_variable_ordinal
        self._binary_label_name = target_variable_binary

        self._cut_point = 5
        self._non_favorable_label_continuous = [5, 6, 7, 8, 9, 10]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)
        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)
