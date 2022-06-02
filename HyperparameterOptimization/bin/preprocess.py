import pandas as pd
import sys
import warnings
from sklearn import preprocessing
import numpy as np
from collections import OrderedDict as od
warnings.filterwarnings("ignore")

sys.path.append("..")
from general.ml import *

"""
This python program does an initial preprocessing on all initial datasets given with the code.
In case you want to use your own dataset, please follow a similar kind of preprocessing for your data in order to have the best possible results
General rules for this preprocessing are:
    -All preprocessed data must be numeric data (for Decision Tree and Logistic Regression to work propperly, as they can't handle categorical data)
    -Numeric data tends to keep the same, with a low number of exceptions
    -Sensitive attribute has to be rewritten as a binary (0 or 1) variable
    -The class attribute (the one we want to predict) has to be also a binary (0 or 1) one
    -The data which is non numeric, has to be processed to transform it into numeric. You can use any approach. As default you can use a simple label encoder
"""

for df_name in ['adult', 'german', 'propublica_recidivism', 'propublica_violent_recidivism', 'ricci']:
    
    df = pd.read_csv('../../data/' + df_name + '.csv', sep = ',')

    if df_name == 'propublica_violent_recidivism' or df_name == 'propublica_recidivism':
        target = df.iloc[:, -1]
        df = df[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'decile_score', 'score_text']]
        df['two_year_recid'] = target

    
    le = preprocessing.LabelEncoder()       #This is a simple label encoding. It doesn't take into account semantical order in labels or semantical similarity or anything like that
                                            #This could be a problem for LR
    for column_name in df.columns[:-1]:
        if df[column_name].dtype == object:
            df[column_name] = df[column_name].astype(str)
            if(column_name == 'race' and df_name == 'adult'):                     #Race is the sensitive attr on adult dataset. We will consider white vs. non-white    
                df[column_name] = np.where(df[column_name] == 'White', 0, 1)      #White is rewritten as 0 and non-white as 1
            elif(column_name == 'sex'):                                           #There are only Male and Female in all cases. we will do the following encoding
                df[column_name] = np.where(df[column_name] == 'Male', 0, 1)       #Male as 0 and Female as 1
            elif(column_name == 'race' and (df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism')):
                df[column_name] = np.where(df[column_name] == 'Caucasian', 0, 1)
            elif(column_name == 'compas_screening_date' or column_name == 'screening_date' or column_name == 'dob'):
                df[column_name] = pd.to_datetime(df[column_name])
                df['year'] = df[column_name].dt.year
                df['month'] = df[column_name].dt.month
                df['day'] = df[column_name].dt.day
                df.drop(column_name, inplace = True, axis = 1)
            elif(column_name == 'Race'):
                df[column_name] = np.where(df[column_name] == 'W', 0, 1)
            elif(column_name == 'score_text'):
                df[column_name] = df[column_name].map(score_text)
            else:                                               #In other case, does the encoding transformation
                df[column_name] = le.fit_transform(df[column_name])
        elif(column_name == 'age' and df_name == 'german'):
            df[column_name] = np.where(df[column_name] > 25, 0, 1)
        else:
            pass

    # Class to predict
    if(df_name == 'adult'):
      df[df.columns[-1]] = np.where(df[df.columns[-1]] == '>50K', 1, 0)
    elif(df_name == 'german'):
      df[df.columns[-1]] = np.where(df[df.columns[-1]] == 1, 0, 1)
    elif(df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism'):
        c = df[df.columns[:-1]].select_dtypes(np.number).columns
        df[c] = df[c].fillna(0)
        df = df.fillna("")
        df[df.columns[-1]] = np.where(df[df.columns[-1]] == 0, 0, 1)
    elif(df_name == 'ricci'):
        df[df.columns[-1]] =  np.where(df[df.columns[-1]] >=  70.000, 0, 1)

    df.to_csv('../../data/' + df_name + '_preproc.csv', sep = ',')

print("Preprocessing execution succesful!\n------------------------------")