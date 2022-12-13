import pandas as pd
import gc

from config import *
from sklearn.model_selection import KFold

data = pd.read_csv(PATH_APPLICATION_TRAIN)
test = pd.read_csv(PATH_APPLICATION_TEST)
prev = pd.read_csv(PATH_PREVIOUS_APPLICATION)

data = data.iloc[0:DEVELOP_DATA_TRAIN_SIZE, :]
test = test.iloc[0:DEVELOP_DATA_TEST_SIZE, :]
prev = prev.iloc[0:DEVELOP_DATA_TEST_SIZE, :]

categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]

for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])

gc.enable()

y_train = data['TARGET']
del data['TARGET']

prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]
for f_ in prev_cat_features:
    prev[f_], _ = pd.factorize(prev[f_])

avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

x_train = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
x_test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

ntrain = x_train.shape[0]
ntest = x_test.shape[0]

excluded_feats = ['SK_ID_CURR']
features = [f_ for f_ in x_train.columns if f_ not in excluded_feats]

x_train = x_train[features]
x_test = x_test[features]

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)