
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
X2 = X.loc[X['train']==0]
y_test = X2['label']
x_test = X2.drop(columns=['train'])
x_test2 = x_test.drop(columns=['label'])

#X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.20, random_state = 42)

clf = RandomForestClassifier(n_estimators = 15, min_samples_split =10, min_samples_leaf = 2, min_weight_fraction_leaf=0,  bootstrap=True, random_state = 42)

clf.fit(x_train, y_train)
y_score = clf.predict(x_test2)

confusion_m = confusion_matrix(y_test,y_score)
print('----------------------------------------------------')
print('')
print('last user as test')
print('')
print('confusion matrix')
print(' ')
print(confusion_m)
print('')
print(classification_report(y_test, y_score))
print('accuracy: ',accuracy_score(y_test, y_score))



path = 'Accelerometer Data 2019-06-28 14-37-42.txt'

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
activityn = np.array(activityn)
y_score= y_score.astype(int)

y_test = y_test.astype(int)
activity = np.array(activity)

print(type(y_test))
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_score, classes= activity, title='Confusion matrix, without normalization')


#path1 = 'Accelerometer Data 2019-06-28 14-00-01.txt'

#son iguales



#path= 'L_pickup1.txt'


#df = df.iloc[300:650]


