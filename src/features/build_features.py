import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


print(os.path.abspath(os.curdir))
os.chdir("../..")
os.chdir(r'data')
print(os.path.abspath(os.curdir))

data_path=r'raw/auto_potential_customers_2018.csv'

def load(data_path):
    qu_2018_raw = pd.read_csv(data_path)
    qu_2018=qu_2018_raw.copy()
    qu_2018['age']=2018-qu_2018['date_of_birth'].str[-4:].astype(int)
    
    return qu_2018

qu_2018=load(data_path)

def age_null(data):
    for i in range(len(data)):
        if pd.isnull(data.agecat.iloc[i]):
            if data.age.iloc[i]>=18 and data.age.iloc[i]<=27:
                data.agecat.iloc[i]=1.0
            elif data.age.iloc[i]>=28 and data.age.iloc[i]<=37:
                data.agecat.iloc[i]=2.0
            elif data.age.iloc[i]>=38 and data.age.iloc[i]<=47:
                data.agecat.iloc[i]=3.0
            elif data.age.iloc[i]>=48 and data.age.iloc[i]<=57:
                data.agecat.iloc[i]=4.0
            elif data.age.iloc[i]>=58 and data.age.iloc[i]<=67:
                data.agecat.iloc[i]=5.0
            elif data.age.iloc[i]>=68:
                data.agecat.iloc[i]=6.0
age_null(qu_2018)

def tra_null(data):
    for i in range(len(data)):
        if pd.isnull(data.traffic_index.iloc[i]):
            if data.area.iloc[i]=='A':
                data.traffic_index.iloc[i]=80.1
            elif data.area.iloc[i]=='B':
                data.traffic_index.iloc[i]=120.1
            elif data.area.iloc[i]=='C':
                data.traffic_index.iloc[i]=133.5
            elif data.area.iloc[i]=='D':
                data.traffic_index.iloc[i]=99.5
            elif data.area.iloc[i]=='E':
                data.traffic_index.iloc[i]=43.5
            elif data.area.iloc[i]=='F':
                data.traffic_index.iloc[i]=116.0
tra_null(qu_2018)

qu_2018.dropna(subset=['credit_score'],inplace=True)

def feature_binary(data):
    data.drop(['quote_number','gender','date_of_birth',
               'agecat','area','veh_body'],axis=1,inplace=True)
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age']
    std = StandardScaler()
    data[dep] = std.fit_transform(data[dep])
      
    data.to_csv('processed/auto_potential_customers_2018_binary.csv')
    return 

def feature_multi(data):
    nosafe=['BUS','CONVT','HDTOP','MCARA','MIBUS','RDSTR','TRUCK']
    data['veh_safe']=1
    data.loc[data['veh_body'].isin(nosafe),'veh_safe']=0

    data.drop(['quote_number','gender','date_of_birth',
               'agecat','area','veh_body'],axis=1,inplace=True)
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age']
    std = StandardScaler()
    data[dep] = std.fit_transform(data[dep])
     
    data.to_csv('processed/auto_potential_customers_2018_multi.csv')
    
    return

def feature_cluster(data):
    data.drop(['quote_number','date_of_birth'],axis=1,inplace=True)

    le = LabelEncoder()
    for i in ['gender']:
        data[i] = le.fit_transform(data[i])

    dep=['agecat', 'credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age']
    std = StandardScaler()
    data[dep] = std.fit_transform(data[dep])

    data = pd.get_dummies(data = data,columns = ['area','veh_body'])
     
    data.to_csv('processed/auto_potential_customers_2018_cluster.csv')
    
    return 