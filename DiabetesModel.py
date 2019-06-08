import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import warnings


def misclass_cleaner(a, b, pred, flag):
    #print(id(a), id(b))
    cl_t = pred.values == b.values
    cl_f = pred.values != b.values

    n_c = max(np.count_nonzero(cl_t), np.count_nonzero(cl_f))
    #print('samples reduced =', n_c)
    if flag:
        if np.count_nonzero(cl_t) > np.count_nonzero(cl_f):
            #print('FFFF')
            a, b = a[cl_t], b[cl_t]
        else:
            #print(('TTTT'))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                temp0 = b == 1
                temp1 = b == 0
                a, b = a[cl_f], b[cl_f]
                b[temp0] = 0
                b[temp1] = 1
        #print('new size = ', np.shape(a))
        return a, b
    else:
        return n_c


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

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, train_size=0.7, test_size=0.3)
n_samples_train, n_features_train = X_train.shape
print('Samples Train =', n_samples_train,
      'Features Train =', n_features_train)

# ______________________ Standardization ______________________

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ___________________________ Decide __________________________

decide = 0.65
n_cleaned = 0
pca = PCA(n_components=2, svd_solver='auto')
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

print(f'''Reducing Features to 2D and Clustering with K-means...
Convergence with 70% left Data.(repeat if more than 30% Data misclassified)
Wait...''')
while n_cleaned < decide*n_samples_train:
    # ____________________________ PCA ____________________________
    pca = PCA(n_components=2, svd_solver='auto')
    pca.fit(X_train)
    pca_train = pca.fit_transform(X_train)
    # __________________________ K-Means __________________________
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(pca_train)

    pred_train = pd.DataFrame(kmeans.labels_, dtype='float64')
    pca_train = pd.DataFrame(pca_train)

    n_cleaned = misclass_cleaner(pca_train, y_train, pred_train, 0)

# __________________________ Cleaning _________________________

print('\nClassification Done!')
X_train = pca.fit_transform(X_train)
pred_train = pd.DataFrame(kmeans.labels_, dtype='float64')
X_train = pd.DataFrame(X_train)
X_train, y_train = misclass_cleaner(X_train, y_train, pred_train, 1)

X_test = pca.fit_transform(X_test)
pred_test = pd.DataFrame(kmeans.predict(X_test), dtype='float64')
X_test = pd.DataFrame(X_test)
X_test, y_test = misclass_cleaner(X_test, y_test, pred_test, 1)
print('\nCleaning Done!')

# __________________________ Visualize _________________________

h = .02
x_min, x_max = X_train[0].min() - 1, X_train[0].max() + 1
y_min, y_max = X_train[1].min() - 1, X_train[1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(X_train[0], X_train[1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='b', zorder=10)
plt.title('K-means clustering on train set of diabetes dataset (PCA-reduced data)\n'
          'Centroids are marked with blue cross')

plt.xticks(())
plt.yticks(())
plt.show()

# ____________________ Logistic Regression ____________________

alphas = np.power(10.0, np.arange(-7, 10))
n_folds = 5
grid = [{'C': alphas}]

lr = LogisticRegression(penalty='l2', random_state=0, max_iter=10)
clf = GridSearchCV(lr, grid, scoring='neg_mean_squared_error', cv=n_folds, return_train_score=True)
print(f'''\nRunning Logistic Regression with different alphas
parameter automatically from the data by internal cross-validation
Wait...''')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    clf.fit(X_train, y_train)
#print(clf.cv_results_)
print('\nTraining Done!')

# _____________________ Cross Validation ______________________

train_err = np.absolute(clf.cv_results_['mean_train_score'])
validate_err = np.absolute(clf.cv_results_['mean_test_score'])
plt.figure(2)
t_err, = plt.semilogx(alphas, train_err, 'b', )
v_err, = plt.semilogx(alphas, validate_err, 'r')
plt.title('Mean Squared Error(MSE) of Logistic Regression\n'
          'with different alphas by internal cross-validation. n_fold = 5')
plt.legend((t_err, v_err), ('Train set', 'Validation Set'))
plt.ylabel('error')
plt.xlabel('alpha')
plt.show()

# _________________________ Evaluation ________________________

y_pred = clf.predict(X_test)

table = pd.DataFrame(columns=['y_test', 'y_pred'])
table['y_test'], table['y_pred'] = y_test['Outcome'], y_pred
#print(table)
print('\nConfusion matrix for test set :')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
