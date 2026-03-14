import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

_dir = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(_dir, 'assignment2train.csv'))

train['DateTime'] = pd.to_datetime(train['DateTime'])
train['hour'] = train['DateTime'].dt.hour
train['minute'] = train['DateTime'].dt.minute
train['dayofweek'] = train['DateTime'].dt.dayofweek

drop_cols = ['id', 'DateTime', 'meal']
feature_cols = [c for c in train.columns if c not in drop_cols]

X_train = train[feature_cols]
y_train = train['meal']

model = DecisionTreeClassifier(
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
modelFit = model.fit(X_train, y_train)

test = pd.read_csv(os.path.join(_dir, 'assignment2test.csv'))
test['DateTime'] = pd.to_datetime(test['DateTime'])
test['hour'] = test['DateTime'].dt.hour
test['minute'] = test['DateTime'].dt.minute
test['dayofweek'] = test['DateTime'].dt.dayofweek

X_test = test[feature_cols]

pred = modelFit.predict(X_test).astype(int).tolist()
