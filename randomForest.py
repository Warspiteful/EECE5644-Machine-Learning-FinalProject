from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder
from random import shuffle
from sklearn.model_selection import RandomizedSearchCV as RSCV

file_path = "data.csv"
df = pd.read_csv(file_path, index_col='year')

file_path = "tracks_features.csv"
df2 = pd.read_csv(file_path, index_col='year')

df.drop(['artists','id', 'name', 'release_date', 'popularity' ], axis = 1, inplace = True)
df2.drop(['artists','id', 'name', 'release_date', 'album_id','artist_ids', 'time_signature', 'track_number', 'disc_number', 'album' ], axis = 1, inplace = True)

df = pd.concat([df,df2])
### Only look at decades from 50s to 10s (2020 not included)
l_drop = np.arange(1921,1950)
l_drop = np.append(l_drop, 2020)
l_drop = np.append(l_drop, 0)
df.drop(labels=l_drop, axis=0, inplace = True)


enc = LabelEncoder()
labels = df.index
standardized_labels = np.array(labels)
enc.fit(df.index.unique())



lmao = df.index
y = enc.transform(standardized_labels)
Y = enc.transform(np.unique(standardized_labels))

y = y//10
Y_decade = np.unique(y)

enc.fit(df['explicit'].unique())
df['explicit'] = enc.transform(df['explicit'])

df.set_index(y, inplace=True)


covariance = df.std()
mean = df.mean()
X = (df-df.mean())/df.std()


X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.30)



# https://towardsdatascience.com/mastering-random-forests-a-comprehensive-guide-51307c129cb1
param_grid = {'n_estimators':np.arange(50,200,15),
              'max_features':np.arange(0.1, 1, 0.1),
              'max_depth': [3, 5, 7, 9],
              'max_samples': [0.3, 0.5, 0.8]}

clf = RSCV(RandomForestClassifier(), param_grid, n_iter = 15).fit(X_train, y_train)
clf = clf.best_estimator_



clf.fit(X_train, y_train)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

feature_imp = pd.Series(clf.feature_importances_, index = df.columns).sort_values(ascending = False)
print(feature_imp)




y_pred = clf.predict(X_test)

scores = {'column': [],
         'importance': []}

labels = df['Cost']
columns = df.drop('Cost', axis = 1).columns()
base_score = clf.score(df[columns], labels)

for col in columns:
	X = df[columns].copy()
	X[col] = shuffle(X[col])
	scores['column'].append(col)
	imp = base_score - clf.score(X, labels)
	scores['importance'].append(imp)

print(pd.DataFrame(scores))

# metrics are used to find accuracy or error
from sklearn import metrics 

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))