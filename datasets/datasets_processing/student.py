import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from datasets_processing._dataset import BaseDataset


class StudentDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'countinuous'
        self._name = 'student'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df = pd.read_csv("original_data/student/student-por.csv", sep=';')

        drop_col = ['school', 'G1', 'G2']
        df.drop(drop_col, axis=1, inplace=True)

        numeric_columns = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
                           "Dalc", "Walc", "health", "absences", "G3"]
        categorical_columns = ["address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
                               "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        cat_data = cat_encoder.fit_transform(df[categorical_columns])
        catnewcols = np.concatenate(
            [[cat] if len(item) == 2 else [cat + '_' + cn for cn in item[1:]] for cat, item in
             zip(categorical_columns, cat_encoder.categories_)]).tolist()

        cat_df = pd.DataFrame(cat_data, columns=catnewcols)

        sex_df = df['sex'].map({'F': 1, 'M': 0})

        dataset = pd.concat([cat_df, sex_df, df[numeric_columns]], axis=1)

        # computing the binary outcome
        dataset['G3_binary'] = dataset['G3'] >= 12
        dataset['G3_binary'] = dataset['G3_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                                       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'sex',
                                       'address', 'famsize', 'Pstatus', 'Mjob_health', 'Mjob_other',
                                       'Mjob_services', 'Mjob_teacher', 'Fjob_health', 'Fjob_other',
                                       'Fjob_services', 'Fjob_teacher', 'reason_home', 'reason_other',
                                       'reason_reputation', 'guardian_mother', 'guardian_other', 'schoolsup',
                                       'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                                       'romantic']

        if self.outcome_type == 'binary':
            self._outcome_label = 'G3_binary'  # [0, 1]
        else:
            self._outcome_label = 'G3'

        # https://jamanetwork.com/journals/jamaneurology/fullarticle/799064#:~:text=Estimates%20for%20the%20UPDRS%20total,and%20response%20to%20therapeutic%20interventions.
        self._favorable_label_binary = [1]
        self._favorable_label_continuous = lambda x: x >= 12

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # female
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'G3'
        self._binary_label_name = 'G3_binary'

        self._cut_point = 12  # is >= all above is 1
        self._non_favorable_label_continuous = lambda x: x < 12

        self._fav_dict = None
        self._nonf_dict = None

        self._middle_fav = None
        self._middle_non_fav = None

