import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('./SVMtrain.csv')
print(data.head(), "\n")


# To define the input and output feature
x = data.drop(['Embarked','PassengerId', 'Sex'],axis=1)
y = data.Embarked
# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

model = lgb.LGBMClassifier()

print(x_train)

print(y_train)
print("aaaaa")
model.fit(x_train, y_train)
print("bbbbb")
y_test_pred = model.predict(x_test)

acc = 0
for i in range(len(y_test)):
    if y_test_pred[i] == y_test.iloc[i]:
        acc += 1

print(y_test_pred, y_test, "\n", acc/len(y_test))



