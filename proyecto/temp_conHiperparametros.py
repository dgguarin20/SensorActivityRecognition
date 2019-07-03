# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

path = 'Jc_pass5.txt'

all_data = pd.DataFrame()

nombre = ['Jc','L','D','X']
activity =['hold','pass','dribble','shoot','pickup']
activityn=[1,2,3,4,5]

label = []


#
df=pd.read_csv(path, header=2,index_col=0)
sns.set(style="whitegrid")
sns.lineplot(data=df)
#
#
for i in range(len(activity)):
    for j in range(len(nombre)):
        m = 3
        if(activity[i] == 'pass'  or activity[i] == 'shoot' ):
            m =5
        if(nombre[j] == 'X'): #solo se tienen 3 trials del usuario X
            m = 3
        for s in range(1,m+1):
            path = nombre[j]+'_'+activity[i]+str(s)+'.txt'
            print(path)
            #df=pd.read_csv(path, header=2)
            df=pd.read_csv(path,header=2)
            if nombre[j] == 'X':
                indices = list(df)
                for d in range(1,5):
                    print('hola')
                    d2 = d+1
                    df[indices[d2]] *= 9.80665

            if activity[i] == 'dribble':
               df=df.iloc[300:900]
            elif activity[i] == 'pass':
                df=df.iloc[300:500]
            elif activity[i] == 'shoot':
                df=df.iloc[:300]
            elif activity[i] == 'pickup':
                df=df.iloc[300:500]
            
            df['label']=activityn[i] 
            arreglo = []
            arreglo.append(df)
            arreglo.append(activity[i])
            label.append(arreglo)
            #plt.figure()
            #plt.title(activity[i]+' '+nombre[j])
            #sns.set(style="whitegrid")
            #sns.lineplot(data=df)

alldata = pd.read_csv('Jc_hold1.txt', header=2)
alldata['label'] = 1
for i in range(len(label)-1):
    d = label[i+1]
    data = d[0]
    

    if(i==0 or i ==9 or i==30):
        valor = alldata['Time (s)'][alldata.index[-1]]
    else:
  
        v = alldata['Time (s)'][alldata.index[-1]]
        
        vv = v.values
        valor = vv[len(vv)-1]

    data['Time (s)']+= valor                                                                                                          
    alldata = pd.concat([alldata, data])
    
labels = alldata['label']
nom = 'Time (s)'
columns =alldata.columns[~alldata.columns.isin(['label', 'Time (s)'])]
filtered_data = alldata[columns].rolling(11).median()
filtered_data['time']=pd.to_datetime(alldata['Time (s)'], unit='s')
filtered_data.index=filtered_data.time
keep = filtered_data.time.dt.microsecond/1000 %400
valorvalor = keep
keep = keep - keep.shift()  < 0
means = filtered_data[columns].rolling('0.4S').mean()[keep]
means.columns = [str(col) + '_mean' for col in means.columns]
variances = filtered_data[columns].rolling('0.4S').var()[keep]


variances.columns = [str(col) + '_var' for col in variances.columns]
labels.index = filtered_data.time
mode_labels = labels.rolling('0.4S').apply(lambda x: mode(x)[0])[keep]
#    
#     #all features
all_features = pd.concat([means, variances], axis=1)
all_features['label'] = mode_labels
all_data = all_features
all_data = all_data.dropna()

y = all_data['label']
X = all_data.drop(columns=['label'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)



#Hyperparameter Tuning

# se buscaran los mejores valores para:
#n_estimators (5-100)
#min_samples_split(2-30)
#min_samples_leaf(1-30)
#min_weight_fraction_leaf(0-0.5) solo se puede estos valores
#bootstrap(True, False (1,0))


max_accuracy = 0

nEstimators = 10
minSamplesSplit = 2
minSamplesLeaf = 1
minWeightFractionLeaf = 0
bootstrap = 0


while nEstimators <= 15:
    while minSamplesSplit <= 30:
        while minSamplesLeaf <=30:
            while minWeightFractionLeaf <=0.5:
                while bootstrap <= 1:
                    clf = RandomForestClassifier(n_estimators=nEstimators, min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf, min_weight_fraction_leaf=minWeightFractionLeaf, bootstrap=True)
                    clf.fit(X_train, y_train)
                    y_score = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_score)
                    if acc > max_accuracy:
                        print("-------------------------------------------------------------")
                        print("n_estimators: " + str(nEstimators))
                        print("min_samples_split: " + str(minSamplesSplit))
                        print("min_samples_leaf: " + str(minSamplesLeaf))
                        print("min_weight_fraction_leaf: " + str(minWeightFractionLeaf))
                        print("boostrap: " + str(bootstrap))
                        print("ACCURACY: " + str(acc))
                        print("-------------------------------------------------------------")
                        max_accuracy = acc
                    bootstrap = bootstrap + 1
                bootstrap = 0
                minWeightFractionLeaf = minWeightFractionLeaf + 0.1
            minWeightFractionLeaf = 0
            minSamplesLeaf = minSamplesLeaf + 1
        minSamplesLeaf = 1
        minSamplesSplit = minSamplesSplit + 1
    minSamplesSplit = 2
    nEstimators = nEstimators + 1




                



##drible 3-9
#pass 1 y 3 JC estan raros



