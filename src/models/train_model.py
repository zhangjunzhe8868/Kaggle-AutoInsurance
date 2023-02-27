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
os.chdir(r'data')
print(os.path.abspath(os.curdir))

data_path=r'raw/auto_policies_2017.csv'

def load(data_path):
    po_2017_raw = pd.read_csv(data_path)
    po_2017=po_2017_raw.copy()
    po_2017['age']=2017-po_2017['date_of_birth'].str[-4:].astype(int)
    
    return po_2017

po_2017=load(data_path)

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
age_null(po_2017)

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
tra_null(po_2017)

po_2017.dropna(subset=['credit_score'],inplace=True)


def model_binary(data):
    data['have_claim']=np.where(data['numclaims']!=0,1,0)
    data.drop(['pol_number', 'pol_eff_dt', 'gender', 'date_of_birth',
               'claim_office', 'numclaims', 'claimcst0', 'annual_premium'],axis=1,inplace=True)
    data.drop(['agecat','area','veh_body'],axis=1,inplace=True)
    
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age']
    std = StandardScaler()
    data[dep] = std.fit_transform(data[dep])
    
    Y = data['have_claim'] 
    X = data.drop('have_claim', axis= 1) 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)
    
    #undersampling majority class using random sampling
    RUS = RandomUnderSampler(random_state = 11)
    us_rs_X,us_rs_Y = RUS.fit_resample(x_train,y_train)
    us_rs_X = pd.DataFrame(data = us_rs_X,columns=dep)
    us_rs_Y = pd.DataFrame(data = us_rs_Y,columns=['have_claim'])
    
    logit  = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
              penalty='l2', random_state=1234, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    logit.fit(us_rs_X, us_rs_Y)
    predict_test = logit.predict(x_test) 
    probabilities_test = logit.predict_proba(x_test) 
    cm=confusion_matrix(y_test,predict_test)
    print('auc:%.3f' %roc_auc_score(y_test, predict_test))
    print("AccuracyScore:%.3f" %accuracy_score(y_test,predict_test))
    print("Precision:%.3f" %precision_score(y_test,predict_test,average='weighted'))
    print("False Alarm Rate:%.3f" %(cm[1][0]/(cm[1][0]+cm[1][1])))
        
    return

def model_multi(data):
    data['have_claim']=np.where(data['numclaims']!=0,1,0)
    data['claim_total']=data['numclaims']*data['claimcst0']
    data['claim_avg_veh_ratio']=data['claimcst0']/(data['veh_value']*10000)
    data['claim_avg_veh_ratio']=np.where(data['claim_avg_veh_ratio']>10000000000,0,data['claim_avg_veh_ratio'])
    
    nosafe=['BUS','CONVT','HDTOP','MCARA','MIBUS','RDSTR','TRUCK']
    data['veh_safe']=1
    data.loc[data['veh_body'].isin(nosafe),'veh_safe']=0
    
    data.drop(['pol_number', 'pol_eff_dt', 'gender', 'date_of_birth',
                   'claim_office', 'numclaims',  'annual_premium',
                   'claim_total', 'claim_total_veh_ratio'],axis=1,inplace=True)
    data.drop(['agecat','area','veh_body'],axis=1,inplace=True)
    
    le = LabelEncoder()
    for i in ['veh_safe']:
        data[i] = le.fit_transform(data[i])
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age','claimcst0']
    std = StandardScaler()
    data[dep] = std.fit_transform(data[dep])
    
    #Independent Variables
    x = data.drop("claimcst0", axis = 1) 
    #Depenedent Variables 
    y = data["claimcst0"]
    
    x_train,x_test, y_train, y_test = train_test_split(x,y, test_size =0.25, random_state = 3)
    
    regressor=RandomForestRegressor()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print('r2: ', r2_score(y_test, y_pred))
    print('rmse: ', sqrt(mean_squared_error(y_test,  y_pred)))

    return 

def model_cluster(data):
    data.drop(['pol_number', 'pol_eff_dt', 'date_of_birth', 'claim_office', 'annual_premium', 
               'agecat', 'area'],axis=1,inplace=True)
    
    #convert binary
    le = LabelEncoder()
    for i in ['gender']:
        data[i] = le.fit_transform(data[i])
    #scaling the numerical data
    dep=['credit_score', 'traffic_index', 'veh_age', 'veh_value', 'age', 'numclaims', 'claimcst0']
    std = StandardScaler()
    data[dep] = std.fit_transform(data[dep])
    
    nosafe=['BUS','CONVT','HDTOP','MCARA','MIBUS','RDSTR','TRUCK']
    data['veh_safe']=1
    data.loc[data['veh_body'].isin(nosafe),'veh_safe']=0
    data.drop('veh_body',axis=1,inplace=True)
    
    data=data.loc[data['has_claim']==1,:]
    data.drop('has_claim',axis=1,inplace=True)
    
    model=KMeans(n_clusters=3)
    model.fit(data)
    labels=model.labels_
    
    data["CLUSTER"]=3
    data.loc[data['has_claim']==1,"CLUSTER"]=labels
    cluster_df=data.groupby("CLUSTER").mean().T
    
    print(os.path.abspath(os.curdir))
    os.chdir("..")
    os.chdir(r'reports/figures')
    print(os.path.abspath(os.curdir))
    
    for i in range(0,5):
        sns.barplot(x=cluster_df[i],y=cluster_df.index)
        #plt.xlim(-0.5,1.5)
        plt.savefig('cluster_'+str(i)+'.png')
    
    return 