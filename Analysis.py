import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import multivariate_normal # MVN not univariate

file_path = "songs_normalize.csv"
df = pd.read_csv(file_path, index_col='year')
enc = LabelEncoder()

labels = df.index
standardized_labels = np.array(labels)
enc.fit(df.index.unique())
aa = df.index.value_counts().sort_index().to_numpy()
priors = aa/len(df.index)
check = np.sum(priors)
class_priors = np.diag(priors)

lmao = df.index
y = enc.transform(standardized_labels)
Y = enc.transform(np.unique(standardized_labels))


df.drop(['artist','song', 'genre' ], axis = 1, inplace = True)

enc.fit(df['explicit'].unique())
df['explicit'] = enc.transform(df['explicit'])
covariance = df.std()
means = df.mean()



X = (df-df.mean())/df.std()

indexer = df.index.values
print(df.head())



columns = df.columns

mu = []

labels = set()
for ax in df.index:
    labels.add(ax)

for index in indexer:
    mu.append([np.mean(X[feature][index]) for feature in df])
mu = np.array(mu)
n = mu.shape[1]
Sigma = []

for index in indexer:
    Sigma.append(np.cov([X[feature][index] for feature in df]))

print(df.head())

C = len(priors)



class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in range(len(Y))])

# Class Posterior
# P(yj | x) = p(x | yj) * P(yj) / p(x)
class_posteriors = class_priors.dot(class_cond_likelihoods)


decisions = np.argmax(class_posteriors, axis=0)+3

sample_class_counts = np.array([sum(y == j) for j in Y])


conf_mat = np.zeros((C, C))
display_mat = np.zeros((C,C))
for i in range(len(Y)): # Each decision option
    for j in range(len(Y)): # Each class label
        ind_ij = np.argwhere((decisions==Y[i]) & (y==Y[j]))
        display_mat[i, j] = len(ind_ij) # Average over class sample count
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j]

print("Confusion matrix:")
print(display_mat)
print(np.sum(display_mat))


