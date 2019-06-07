import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import confusion_matrix

# _______________________ Preparing Data ______________________

Data = pd.read_csv('Datasets/diabetes.csv')
Data = Data.astype('float64')
print(Data.head())
X = Data[['Pregnancies',
          'Glucose',
          'BloodPressure',
          'SkinThickness',
          'Insulin',
          'BMI',
          'DiabetesPedigreeFunction',
          'Age']]
y = Data[['Outcome']]

# ___________________________ Split ___________________________

X_train, X_test, y_train, y_test = train_test_split(X, y)

# ______________________ Standardization ______________________

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)
print('StandardScaler X =\n', X_train, X_train.shape)

# ____________________________ PCA ____________________________

n_features = 8
pca = PCA(n_components=1, svd_solver='auto')
pca.fit(X_train)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
print('pca X =\n', X, X.shape)

# __________________________ K-Means __________________________

k_means = KMeans(n_clusters=10)
k_means.fit(X_train)
X_train = k_means.labels_
X_train = X_train.reshape(-1, 1)

X_test = k_means.predict(X_test)
X_test = X_test.reshape(-1, 1)

print('clustred X =\n', X_train.reshape(1, -1), X_train.shape)
print('clustred X =\n', X_test.reshape(1, -1), X_test.shape)
# ____________________ Logistic Regression ____________________

clf = LogisticRegression(solver='lbfgs')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    clf.fit(X_train, y_train)

# _________________________ Evaluation ________________________

y_pred = clf.predict(X_test)
y_pred = y_pred.reshape(-1, 1)

table = pd.DataFrame(columns=['y_test', 'y_pred'])
table['y_test'] = y_test['Outcome']
table['y_pred'] = y_pred
print(table)
print(confusion_matrix(y_test, y_pred))

y_pred = clf.predict(X_train)
print(confusion_matrix(y_train, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred))
