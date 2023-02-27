import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.cluster import KMeans

print(os.path.abspath(os.curdir))
os.chdir("../..")
os.chdir(r'po_2017')
print(os.path.abspath(os.curdir))

po_2017_path=r'raw/auto_policies_2017.csv'

def load(po_2017_path):
    po_2017_raw = pd.read_csv(po_2017_path)
    po_2017=po_2017_raw.copy()
    po_2017['age']=2017-po_2017['date_of_birth'].str[-4:].astype(int)
    
    return po_2017

po_2017=load(po_2017_path)

def age_null(po_2017):
    for i in range(len(po_2017)):
        if pd.isnull(po_2017.agecat.iloc[i]):
            if po_2017.age.iloc[i]>=18 and po_2017.age.iloc[i]<=27:
                po_2017.agecat.iloc[i]=1.0
            elif po_2017.age.iloc[i]>=28 and po_2017.age.iloc[i]<=37:
                po_2017.agecat.iloc[i]=2.0
            elif po_2017.age.iloc[i]>=38 and po_2017.age.iloc[i]<=47:
                po_2017.agecat.iloc[i]=3.0
            elif po_2017.age.iloc[i]>=48 and po_2017.age.iloc[i]<=57:
                po_2017.agecat.iloc[i]=4.0
            elif po_2017.age.iloc[i]>=58 and po_2017.age.iloc[i]<=67:
                po_2017.agecat.iloc[i]=5.0
            elif po_2017.age.iloc[i]>=68:
                po_2017.agecat.iloc[i]=6.0
age_null(po_2017)

def tra_null(po_2017):
    for i in range(len(po_2017)):
        if pd.isnull(po_2017.traffic_index.iloc[i]):
            if po_2017.area.iloc[i]=='A':
                po_2017.traffic_index.iloc[i]=80.1
            elif po_2017.area.iloc[i]=='B':
                po_2017.traffic_index.iloc[i]=120.1
            elif po_2017.area.iloc[i]=='C':
                po_2017.traffic_index.iloc[i]=133.5
            elif po_2017.area.iloc[i]=='D':
                po_2017.traffic_index.iloc[i]=99.5
            elif po_2017.area.iloc[i]=='E':
                po_2017.traffic_index.iloc[i]=43.5
            elif po_2017.area.iloc[i]=='F':
                po_2017.traffic_index.iloc[i]=116.0
tra_null(po_2017)

po_2017.dropna(subset=['credit_score'],inplace=True)


def model_binary(data):
    po_2017['have_claim']=np.where(po_2017['numclaims']!=0,1,0)
    po_2017.drop(['pol_number', 'pol_eff_dt', 'gender', 'date_of_birth',
               'claim_office', 'numclaims', 'claimcst0', 'annual_premium'],axis=1,inplace=True)
    po_2017.drop(['agecat','area','veh_body'],axis=1,inplace=True)
    
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age']
    std = StandardScaler()
    po_2017[dep] = std.fit_transform(po_2017[dep])
    
    Y = po_2017['have_claim'] 
    X = po_2017.drop('have_claim', axis= 1) 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)
    
    #undersampling majority class using random sampling
    RUS = RandomUnderSampler(random_state = 11)
    us_rs_X,us_rs_Y = RUS.fit_resample(x_train,y_train)
    us_rs_X = pd.po_2017Frame(po_2017 = us_rs_X,columns=dep)
    us_rs_Y = pd.po_2017Frame(po_2017 = us_rs_Y,columns=['have_claim'])
    
    logit  = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
              penalty='l2', random_state=1234, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    logit.fit(us_rs_X, us_rs_Y)
    
    predict_test = logit.predict(data) 
    probabilities_test = logit.predict_proba(data)  
        
    return predict_test

def model_multi(data):
    po_2017['have_claim']=np.where(po_2017['numclaims']!=0,1,0)
    po_2017['claim_total']=po_2017['numclaims']*po_2017['claimcst0']
    po_2017['claim_avg_veh_ratio']=po_2017['claimcst0']/(po_2017['veh_value']*10000)
    po_2017['claim_avg_veh_ratio']=np.where(po_2017['claim_avg_veh_ratio']>10000000000,0,po_2017['claim_avg_veh_ratio'])
    
    nosafe=['BUS','CONVT','HDTOP','MCARA','MIBUS','RDSTR','TRUCK']
    po_2017['veh_safe']=1
    po_2017.loc[po_2017['veh_body'].isin(nosafe),'veh_safe']=0
    
    po_2017.drop(['pol_number', 'pol_eff_dt', 'gender', 'date_of_birth',
                   'claim_office', 'numclaims', 'annual_premium',
                   'claim_total',  'claim_avg_veh_ratio'],axis=1,inplace=True)
    po_2017.drop(['agecat','area','veh_body'],axis=1,inplace=True)
    
    le = LabelEncoder()
    for i in ['veh_safe']:
        po_2017[i] = le.fit_transform(po_2017[i])
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age','claimcst0']
    std = StandardScaler()
    po_2017[dep] = std.fit_transform(po_2017[dep])
    
    #Independent Variables
    x = po_2017.drop("claim_avg", axis = 1) 
    #Depenedent Variables 
    y = po_2017["claim_avg"]
    
    x_train,x_test, y_train, y_test = train_test_split(x,y, test_size =0.25, random_state = 3)
    
    regressor=RandomForestRegressor()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(data)

    return y_pred

def model_cluster(data):
    
    model=KMeans(n_clusters=3)
    model.fit(data)
    labels=model.labels_
    
    data["CLUSTER"]=labels
    cluster_df=data.groupby("CLUSTER").mean().T
    
    
    return  cluster_df