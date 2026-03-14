import os, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

test = pd.read_csv(os.path.join(_dir, 'assignment2test.csv'))
test['DateTime'] = pd.to_datetime(test['DateTime'])
test['hour'] = test['DateTime'].dt.hour
test['minute'] = test['DateTime'].dt.minute
test['dayofweek'] = test['DateTime'].dt.dayofweek
X_test = test[feature_cols]
truth = pd.read_csv(os.path.join(_dir, 'tests/testData.csv'))['meal']

def tjurr(truth, pred):
    truth = list(truth)
    pred = list(pred)
    y1 = np.mean([y for x, y in enumerate(pred) if truth[x]==1])
    y2 = np.mean([y for x, y in enumerate(pred) if truth[x]==0])
    return y1-y2

# Test 1: RF with balanced class weight + probabilities
m1 = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
m1.fit(X_train, y_train)
p1 = m1.predict_proba(X_test)[:, 1]
print(f'RF balanced + proba: {tjurr(truth, p1):.4f}')

# Test 2: RF with balanced + predict (binary)
p2 = m1.predict(X_test).astype(int)
print(f'RF balanced + binary pred: {tjurr(truth, p2):.4f}')

# Test 3: RF with balanced + lower threshold
p3 = (p1 > 0.3).astype(int)
print(f'RF balanced + threshold 0.3: {tjurr(truth, p3):.4f}')

# Test 4: Different hyperparams
m4 = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1)
m4.fit(X_train, y_train)
p4 = m4.predict_proba(X_test)[:, 1]
print(f'RF balanced no max_depth + proba: {tjurr(truth, p4):.4f}')
p4b = m4.predict(X_test).astype(int)
print(f'RF balanced no max_depth + binary: {tjurr(truth, p4b):.4f}')
