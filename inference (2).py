from models import AltModel

import numpy as np
import torch
import sklearn
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.optim import AdamW, Adam
from torchmetrics import AUROC
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from torch.autograd import Variable
from torch.optim import lr_scheduler

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



transactions = pd.read_csv(r"./data/transactions.csv")
train = pd.read_csv(r"./data/train.csv")
test = pd.read_csv(r"./data/test.csv")
mcc = pd.read_csv(r'./data/mcc_codes.csv', sep = ';')
trans_types = pd.read_csv(r'./data/trans_types.csv', sep = ';')

le_mcc = LabelEncoder()
le_tt = LabelEncoder()
le_tc = LabelEncoder()



def feature_create(train_bool = True):
    '''
    Необходимо перед вызовом функции создать таблицы transactions, train, test, mcc, trans_types
    '''
    # 1. Encode
    if train_bool:
        x = pd.merge(transactions, train, on=['client_id'])
        x['mcc_code'] = le_mcc.fit_transform(x['mcc_code'])
        
        x['trans_type'] = le_tt.fit_transform(x['trans_type'])
        
        x['trans_city'] = le_tc.fit_transform(x['trans_city'])
        
        
        gender_dict_train = dict(x[['client_id', 'gender']].values)
        # Аррау айдишников из трэйна
        train_ids = list(gender_dict_train.keys())
        # Аррау полов из трэйна
        train_genders = list(gender_dict_train.values())
        
        mcc_v_train = {key: [0]*len(mcc) for key in train_ids}
        trans_types_v_train = {key: [0]*len(trans_types) for key in train_ids}
        incomes_train = {key: [0]*len(mcc) for key in train_ids}
        outcomes_train = {key: [0]*len(mcc) for key in train_ids}
        cities_train = {key: [0]*10 for key in train_ids}
        incomes_at_hour_train = {key: [0]*24 for key in train_ids}
        outcomes_at_hour_train = {key: [0]*24 for key in train_ids}
        
        x['H'] = x['trans_time'].str[-8:-6].astype('int')
            
        incomes_transactions_train = x[x['amount'] > 0].reset_index(drop=True)
        outcomes_transactions_train = x[x['amount'] < 0].reset_index(drop=True)

        for index in range(len(x)):
            mcc_v_train[x.iloc[index, 0]][x.iloc[index, 2]] += 1
            trans_types_v_train[x.iloc[index, 0]][x.iloc[index, 3]] += 1
            cities_train[x.iloc[index, 0]][x.iloc[index, 6]] = 1
        
        for index in range(len(incomes_transactions_train)):
            incomes_train[incomes_transactions_train.iloc[index, 0]][incomes_transactions_train.iloc[index, 2]] += incomes_transactions_train.iloc[index, 4]
            incomes_at_hour_train[incomes_transactions_train.iloc[index, 0]][incomes_transactions_train.iloc[index, 9]] += incomes_transactions_train.iloc[index, 4]
            
        for index in range(len(outcomes_transactions_train)):
            outcomes_train[outcomes_transactions_train.iloc[index, 0]][outcomes_transactions_train.iloc[index, 2]] += outcomes_transactions_train.iloc[index, 4]
            outcomes_at_hour_train[outcomes_transactions_train.iloc[index, 0]][outcomes_transactions_train.iloc[index, 9]] += outcomes_transactions_train.iloc[index, 4]
              
        return pd.DataFrame(mcc_v_train).T, pd.DataFrame(trans_types_v_train).T, pd.DataFrame(incomes_train).T, pd.DataFrame(outcomes_train).T, pd.DataFrame(cities_train).T, pd.DataFrame(incomes_at_hour_train).T, pd.DataFrame(outcomes_at_hour_train).T
    
    else:
        
        y = pd.merge(transactions, train, on=['client_id'])
        y['mcc_code'] =  le_mcc.fit_transform(y['mcc_code'])
        
        y['trans_type'] = le_tt.fit_transform(y['trans_type'])
        
        y['trans_city'] = le_tc.fit_transform(y['trans_city'])
        
        
        x = pd.merge(transactions, test, on=['client_id'])
        x['mcc_code'] = le_mcc.transform(x['mcc_code'])
        x['trans_type'] = le_tt.transform(x['trans_type'])
        x['trans_city'] = le_tc.transform(x['trans_city'])
                          
        test_ids = list(x['client_id'].unique())
                          
        mcc_v_test = {key: [0]*len(mcc) for key in test_ids}
        trans_types_v_test = {key: [0]*len(trans_types) for key in test_ids}
        incomes_test = {key: [0]*len(mcc) for key in test_ids}
        outcomes_test = {key: [0]*len(mcc) for key in test_ids}
        cities_test = {key: [0]*10 for key in test_ids}
        incomes_at_hour_test = {key: [0]*24 for key in test_ids}
        outcomes_at_hour_test = {key: [0]*24 for key in test_ids}
        
        x['H'] = x['trans_time'].str[-8:-6].astype('int')
            
        incomes_transactions_test = x[x['amount'] > 0].reset_index(drop=True)
        outcomes_transactions_test = x[x['amount'] < 0].reset_index(drop=True)
            
        for index in range(len(x)):
            mcc_v_test[x.iloc[index, 0]][x.iloc[index, 2]] += 1
            trans_types_v_test[x.iloc[index, 0]][x.iloc[index, 3]] += 1
            cities_test[x.iloc[index, 0]][x.iloc[index, 6]] = 1
        
        for index in range(len(incomes_transactions_test)):
            incomes_test[incomes_transactions_test.iloc[index, 0]][incomes_transactions_test.iloc[index, 2]] += incomes_transactions_test.iloc[index, 4]
            incomes_at_hour_test[incomes_transactions_test.iloc[index, 0]][incomes_transactions_test.iloc[index, 8]] += incomes_transactions_test.iloc[index, 4]
            
        for index in range(len(outcomes_transactions_test)):
            outcomes_test[outcomes_transactions_test.iloc[index, 0]][outcomes_transactions_test.iloc[index, 2]] += outcomes_transactions_test.iloc[index, 4]
            outcomes_at_hour_test[outcomes_transactions_test.iloc[index, 0]][outcomes_transactions_test.iloc[index, 8]] += outcomes_transactions_test.iloc[index, 4]
                          
        return pd.DataFrame(mcc_v_test).T, pd.DataFrame(trans_types_v_test).T, pd.DataFrame(incomes_test).T, pd.DataFrame(outcomes_test).T, pd.DataFrame(cities_test).T, pd.DataFrame(incomes_at_hour_test).T, pd.DataFrame(outcomes_at_hour_test).T

    


mcc_freq, ttp_freq, inc, out, cities, inc_h, out_h = feature_create(False)  


class TestData(pl.LightningDataModule):
    def __init__(self, mcc, ttp, inc, out, inc_h, out_h, cities):
        super().__init__()
        self.mcc = torch.tensor(mcc).to(torch.float32)
        self.ttp = torch.tensor(ttp).to(torch.float32)
        self.inc = torch.tensor(inc).to(torch.float32)
        self.out = torch.tensor(out).to(torch.float32)
        self.inc_h = torch.tensor(inc_h).to(torch.float32)
        self.out_h = torch.tensor(out_h).to(torch.float32)
        self.cities = torch.tensor(cities).to(torch.float32)
        self.features = [self.mcc, self.ttp, self.inc, self.out, self.inc_h, self.out_h, self.cities]
        
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        dataset = TensorDataset(*self.features)
        self.test = dataset
    
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=len(self.test))
    
    
def preprocess(data):
    eps = 1e-5
    data = data.T
    scale = np.abs(data.sum().values)
    data = (data / (scale + eps)).T
    return data


mcc_freq, ttp_freq, inc, out, inc_h, out_h = map(preprocess, [mcc_freq, ttp_freq, inc, out, inc_h, out_h])



client_id = test.client_id.values


data = TestData(mcc_freq.values, ttp_freq.values, 
                   inc.values, out.values, inc_h.values, out_h.values, cities.values)
data.setup()

model = AltModel.load_from_checkpoint("./epoch=299-step=3600.ckpt", map_location='cpu')
model.eval()
model.cpu()

for inputs in data.test_dataloader():
    mcc_freq, ttp_freq, inc, out, inc_h, out_h, cities = inputs
    with torch.no_grad():
        preds = F.sigmoid(model(mcc_freq, ttp_freq, inc, out, inc_h, out_h, cities))
        
preds = preds.detach().numpy()

submission = pd.DataFrame({
    'client_id' : client_id,
    'probability' : preds[:, 0]
})

submission.to_csv('result.csv')