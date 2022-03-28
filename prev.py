# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('prevOficial.csv')
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 3].values

dfX = pd.DataFrame(X)
dfy = pd.DataFrame(y)
#fica com a dimensao certa
y = np.array([y]).reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train1[:, 0:3])
X_test = sc.transform(X_test1[:, 0:3])

#%%
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
for i in range (2,4):
 classifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0)
 classifier.fit(X_train, y_train.ravel())
 from sklearn.metrics import accuracy_score
 y_pred = classifier.predict(X_test)
 print("Número de estimadores",i,"\n Accuracy: ",accuracy_score(y_test, y_pred))

#%%
#previsao
y_pred = classifier.predict(X_test)
print("N_teste:",len(y_pred))
X_test = sc.inverse_transform(X_test) 
for i in range (0, len(y_pred)):
    if(y_pred[i] == y_test[i][0]):
        print("IGUAL")
    print("Prev(",i,"):",y_pred[i],"Original",y_test[i][0],X_test1[i])


#%%        
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix")
print(cm)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Recall  : ",recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ",f1_score(y_test, y_pred, average='macro'))

#%% 
# # Visualising the Test set results
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range (len(X)):
    if(y[i]==0):
        ax.scatter(X[i,0], X[i,1], X[i,2], c='r', marker='o')
    if(y[i]==1):
        ax.scatter(X[i,0], X[i,1], X[i,2], c='b', marker='o')
    if(y[i]==3):
        ax.scatter(X[i,0], X[i,1], X[i,2], c='g', marker='o')
plt.title('Random Forest Classification (Test set)')
ax.set_xlabel('Time 1 - Chances')
ax.set_ylabel('Time 2 - Chances')
ax.set_zlabel('Previsão Vitória')
plt.legend()

#%%
# Predicting a new result
a=[]

X_predFull = pd.read_csv('prox_rod14.csv').fillna(0)
X_predFull = X_predFull.iloc[:, 0:6].values
X_pred = sc.transform(X_predFull[:, 0:3])
y_pred = classifier.predict(X_pred)
for i in range (len(X_pred)):
    print("Pontos Previstos: ",y_pred[i],X_predFull[i])

