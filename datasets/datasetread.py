import pandas as pd
import os
import seaborn as sns
import plotly.express as px

dict_outcomes = {
    'academic': 'atd',
    'adult': 'income',
    'arrhythmia': 'arrhythmia',
    'bank': 'Subscribed',
    'catalunya': 'recid',
    'compas': 'score',
    'credit': 'NoDefault',
    'crime': 'ViolentCrimesPerPop',
    'default': 'default',
    'diabetes-w': 'Outcome',
    'diabetes': 'readmitted',
    'drugs': 'Coke',
    'dutch': 'status',
    'german': 'Label',
    'heart': 'class',
    'hrs': 'score',
    'insurance': 'charges',
    'kdd-census': 'Label',
    'lsat':'ugpa',
    'nursery': 'class',
    'obesity': 'NObeyesdad',
    'older-adults': 'mistakes',
    'oulad': 'Grade',
    'parkinson': 'total_UPDRS',
    'ricci': 'Combine',
    'singles': 'income',
    'student': 'G3',
    'tic': 'income',
    'wine': 'quality',
    'synthetic-athlete': 'Label',
    'synthetic-disease': 'Label',
    'toy': 'Label'
}

dict_protected = {
    'academic': 'ge',
    'adult': 'Race',
    'arrhythmia': 'sex',
    'bank': 'AgeGroup',
    'catalunya': 'foreigner',
    'compas': 'race',
    'credit': 'sex',
    'crime': 'race',
    'default': 'SEX',
    'diabetes-w': 'Age',
    'diabetes': 'Sex',
    'drugs': 'Gender',
    'dutch': 'Sex',
    'german': 'Sex',
    'heart': 'Sex',
    'hrs': 'gender',
    'insurance': 'sex',
    'kdd-census': 'Sex',
    'lsat':'race',
    'nursery': 'finance',
    'obesity': 'Gender',
    'older-adults': 'sex',
    'oulad': 'Sex',
    'parkinson': 'sex',
    'ricci': 'Race',
    'singles': 'sex',
    'student': 'sex',
    'tic': 'religion',
    'wine': 'color',
    'synthetic-athlete': 'Sex',
    'synthetic-disease': 'Age',
    'toy': 'sst'
}


selected = ['adult', 'compas', 'diabetes', 'dutch', 'german', 'insurance', 'obesity', 'parkinson', 'ricci', 'student']

datasets_path = '../datasets/data'
x = []
y = []
names = []
sizes = []
for element in os.listdir(datasets_path):
    element_path = os.path.join(datasets_path, element)
    if os.path.isfile(element_path) and element.split('.')[0] in selected:
        data = pd.read_csv(element_path)                            # Read the dataframe
        if element == 'compas.csv':
            data = data.drop('decile_score', axis=1)
        else:
            data = data.drop(dict_outcomes[element[:-4]], axis=1)
        print(dict_outcomes[element[:-4]])
        if dict_outcomes[element[:-4]] + '_binary' in data.columns:
            data['binary_' + dict_outcomes[element[:-4]]] = data[dict_outcomes[element[:-4]] + '_binary']
            data = data.drop(dict_outcomes[element[:-4]] + '_binary', axis=1)
        
        print("----------------")
        print("Dataset name:", element)                             # Basic dataframe information
        #print("Datset's first rows:", data.head(5))
        print("Dataset shape", data.shape)
        #print("Dataset columns:", data.columns)
        x.append(data.shape[0])                                     # For representation purposes
        y.append(data.shape[1])
        names.append(element[:-4])
        sizes.append(1)
        y_var = data['binary_' + dict_outcomes[element[:-4]]]       # Read output variable
        #print("Output attribute:", y_var)
        #print("Output attribute possible values:", y_var.unique())
        assert(len(y_var.unique()) == 2 and 0 in y_var.unique() and 1 in y_var.unique())
        prot_var = data[dict_protected[element[:-4]]]                    # Read protected variable
        print(data.columns)
        print("Protected attribute:", prot_var)
        #print("Protected attribute possible values:", prot_var.unique())
        assert(len(y_var.unique()) == 2 and 0 in prot_var.unique() and 1 in prot_var.unique())


plot_df = pd.DataFrame({'Number of rows': x, 'Number of columns': y, 'Name': names, 'sizes':sizes})
print(plot_df)
fig = px.scatter(plot_df, x = 'Number of rows', y = 'Number of columns', text='Name', hover_data=['Name'])
fig.update_traces(textposition=['top center', 'top center', 'top center', 'top center', 'top center', 'top center', 'bottom center', 'top center', 'top center', 'bottom center'])
fig.update_layout(
    title={
        'text': "Dimensions of the datasets",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(marker=dict(size=10, opacity=0.5))
fig.write_image("./dimensions.pdf")