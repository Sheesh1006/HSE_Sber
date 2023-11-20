import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib


mcc_freq = pd.read_csv("mcc_frequency.csv")
ttp_freq = pd.read_csv("trans_types_frequency.csv")
incomes  = pd.read_csv("Incomes.csv")
outcomes = pd.read_csv("Outcomes.csv")
genders  = pd.read_csv("genders.csv")
mcc_freq = mcc_freq.drop("Unnamed: 0", axis=1)
mcc_freq /= mcc_freq.sum().values
mcc_freq = mcc_freq.transpose()

def preprocess(data):
    data = data.set_index("Unnamed: 0").T
    scale = np.abs(data.sum().values)
    data = (data / scale).T
    return data
genders = preprocess(genders)
ttp_freq = preprocess(ttp_freq)
incomes  = preprocess(incomes )
outcomes = preprocess(outcomes)

df_all = pd.concat([ttp_freq,incomes,outcomes,mcc_freq], axis =1)

X = df_all.fillna(0).to_numpy()
y = genders.fillna(0).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_df = pd.DataFrame(y_train,columns = ['target'])
y_test_df = pd.DataFrame(y_test,columns = ['target'])
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
df_train = pd.concat([X_train_df,y_train_df], axis =1)
df_test = pd.concat([X_test_df,y_test_df], axis =1)
from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
roles = {
    'target': 'target'
}
task = Task('binary')
# Создание экземпляра модели LightAutoML
automl = TabularAutoML(task=task, timeout=3600)
df_train.columns = df_train.columns.astype(str)
df_test.columns = df_test.columns.astype(str)
# Обучение модели
out_of_fold_predictions = automl.fit_predict(df_train, roles = roles, verbose = 1)
test_predictions = automl.predict(df_test)
joblib.dump(automl, 'model.pkl')