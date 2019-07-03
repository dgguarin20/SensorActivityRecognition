


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

path = 'Accelerometer Data 2019-06-28 14-37-42.txt'

#path1 = 'Accelerometer Data 2019-06-28 14-00-01.txt'

#son iguales



#path= 'L_pickup1.txt'

df=pd.read_csv(path, header=2,index_col=0)
#df = df.iloc[300:650]


lista = list(df)
for i in range(0,4):
    df[lista[i]]*=9.80665

df0=df.iloc[0:375]  #caminando a su posicion
df1=df.iloc[375:980]#hold
df1['label'] = 1
df1_5=df.iloc[980:1100]#cambiar de posicion 
df2 =df.iloc[1100:1300] #pass
df2['label'] = 2
df2_5 =df.iloc[1300:1500]#recbir
df2_7 = df.iloc[1500:1670]#hold chiquito o none
df2_7['label'] = 1
df3=df.iloc[1670:1900] #shoot
df3['label'] = 4
df3_5=df.iloc[1900:2300] #esperar el balon
df3_6=df.iloc[2300:2400] #recibir
df3_7=df.iloc[2400:2500] #hold
df3_7['label'] = 1
df5=df.iloc[2500:3150]#dribble CHECK
df5['label'] = 3
df5_5=df.iloc[3150:3450]#hold
df5_5['label'] = 1
df5_6=df.iloc[3450:3600] #bajar
df6 = df.iloc[3600:3840] #pickup
df6['label'] = 5
df6_5 = df.iloc[3840:3900] #se movio mnucho .superior no muy claro

df40 =df.iloc[3900:4050] #hold after pickup
df60=df.iloc[4200:4450]  #alzada cuasi shoot del final
df100=df.iloc[4450:5200]#not in video
sns.set(style="whitegrid")
#sns.lineplot(data=df30[' R (g)'])
unido =  pd.DataFrame()
unido = pd.concat([unido, df1])
unido = pd.concat([unido, df2])
unido = pd.concat([unido, df2_7])
unido = pd.concat([unido, df3])
unido = pd.concat([unido, df3_7])
unido = pd.concat([unido, df5])
unido = pd.concat([unido, df5_5])
unido = pd.concat([unido, df6])

sns.lineplot(data=df40)




#print(df[' Z (g)'].max())
#print(df30[' R (g)'].idxmax())