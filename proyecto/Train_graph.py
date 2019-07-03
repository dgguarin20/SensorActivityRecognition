
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

path = 'Jc_pass5.txt'

all_data = pd.DataFrame()

nombre = ['Jc','L','D','X']
activity =['hold','pass','dribble','shoot','pickup']
activityn=[0,1,2,3,4]

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
        if(activity[i] == 'pass' or activity[i] == 'shoot'):
            m =5
        if(nombre[j] == 'X'): #solo se tienen 3 trials del usuario X
            m = 3
        for s in range(1,m+1):
            path = nombre[j]+'_'+activity[i]+str(s)+'.txt'
            #df=pd.read_csv(path, header=2)
            df=pd.read_csv(path,header=2)
            if nombre[j] == 'X':
                indices = list(df)
                for d in range(1,5):
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
            df['train']= 1
            arreglo = []
            arreglo.append(df)
            arreglo.append(activity[i])
            label.append(arreglo)
            #plt.figure()
            #plt.title(activity[i]+' '+nombre[j])
            #sns.set(style="whitegrid")
            #sns.lineplot(data=df)

path = 'Accelerometer Data 2019-06-28 14-37-42.txt'


df=pd.read_csv(path, header=2)


lista = list(df)
for i in range(0,4):
    df[lista[i]]*=9.80665

df0=df.iloc[0:375]  #caminando a su posicion
df1=df.iloc[375:980]#hold
df1['label'] = 0
df1_5=df.iloc[980:1100]#cambiar de posicion 
df2 =df.iloc[1100:1300] #pass
df2['label'] = 1
df2_5 =df.iloc[1300:1500]#recbir
df2_7 = df.iloc[1500:1670]#hold chiquito o none
df2_7['label'] = 0
df3=df.iloc[1670:1900] #shoot
df3['label'] = 3
df3_5=df.iloc[1900:2300] #esperar el balon
df3_6=df.iloc[2300:2400] #recibir
df3_7=df.iloc[2400:2500] #hold
df3_7['label'] = 0
df5=df.iloc[2500:3150]#dribble CHECK
df5['label'] = 2
df5_5=df.iloc[3150:3450]#hold
df5_5['label'] = 0
df5_6=df.iloc[3450:3600] #bajar
df6 = df.iloc[3600:3840] #pickup
df6['label'] = 4
df6_5 = df.iloc[3840:3900] #se movio mnucho .superior no muy claro

df40 =df.iloc[3900:4050] #hold after pickup
df60=df.iloc[4200:4450]  #alzada cuasi shoot del final
df100=df.iloc[4450:5200]#not in video
sns.set(style="whitegrid")
#sns.lineplot(data=df30[' R (g)'])
unido =  pd.DataFrame()
unido = pd.concat([unido, df1])
unido = pd.concat([unido, df2])

unido = pd.concat([unido, df3])

unido = pd.concat([unido, df5])
unido = pd.concat([unido, df5_5])
unido = pd.concat([unido, df6])

valores = unido.values

unido['train'] = 0

arreglo = []
#
arreglo.append(unido)
arreglo.append(activity[i])
label.append(arreglo)
#


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
trainii = alldata['train']
nom = 'Time (s)'
columns =alldata.columns[~alldata.columns.isin(['label', 'Time (s)','train'])]
filtered_data = alldata[columns].rolling(11).median()
filtered_data['time']=pd.to_datetime(alldata['Time (s)'], unit='s')
filtered_data.index=filtered_data.time
keep = filtered_data.time.dt.microsecond/1000 %200
valorvalor = keep
keep = keep - keep.shift()  < 0
means = filtered_data[columns].rolling('0.4S').mean()[keep]
means.columns = [str(col) + '_mean' for col in means.columns]
variances = filtered_data[columns].rolling('0.4S').var()[keep]


variances.columns = [str(col) + '_var' for col in variances.columns]
labels.index = filtered_data.time
mode_labels = labels.rolling('0.4S').apply(lambda x: mode(x)[0])[keep]

trainii.index = filtered_data.time
mode_trainii = trainii.rolling('0.4S').apply(lambda x: mode(x)[0])[keep]
#    
#     #all features
all_features = pd.concat([means, variances], axis=1)
all_features['label'] = mode_labels
all_features['train'] = mode_trainii

all_data = all_features
all_data = all_data.dropna()


X = all_data
X1 = X.loc[X['train']==1]
y_train = X1['label']
x_train = X1.drop(columns=['label','train'])


X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.20, random_state = 42)

clf = RandomForestClassifier(n_estimators = 15, min_samples_split =10, min_samples_leaf = 2, min_weight_fraction_leaf=0,  bootstrap=True, random_state = 42)

clf.fit(X_train, y_train)
y_score = clf.predict(X_test)


LLAVES=y_test
R=X2.index

n=(W.size)-1
from tkinter import *
from time import sleep

root = Tk()
root.title("ventana")
root.geometry('500x500')
var = StringVar()
var1 = StringVar()

var.set('hello')
var1.set('hello')

l = Label(root, textvariable = var,font=("Courier", 44))
l1 = Label(root, textvariable = var1,font=("Courier", 44))
l1.configure(background='aquamarine')
l1.configure(background='red')


l.pack()
l1.pack()
for i in range(1000):
    sleep(0.0375) # Need this to slow the changes down
      #if i==0:
      #var.set('goodbye' if i%2 else 'hello')  
     # else: 
    #var.set('goodbye'if i%2 else str(activity[int(LLAVES[i]-1)]))
    var.set(str(activity[int(LLAVES[i]-1)]) if i%2 else str(activity[int(LLAVES[i]-1)]))
    var1.set(str(activity[int(y_score[i]-1)]) if i%3000000 else str(activity[int(y_score[i]-1)]))
    l1.configure(background='aquamarine') if int(y_score[i]-1)==int(LLAVES[i]-1) else l1.configure(background='red')

    
    root.update_idletasks()


#path1 = 'Accelerometer Data 2019-06-28 14-00-01.txt'

#son iguales



#path= 'L_pickup1.txt'


#df = df.iloc[300:650]


