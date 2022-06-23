from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

file_path = "data.csv"
df = pd.read_csv(file_path, index_col='year')

df.drop(['artists','id', 'name', 'release_date', 'popularity' ], axis = 1, inplace = True)

### Only look at decades from 50s to 10s (2020 not included)
l_drop = np.arange(1921,1950)
l_drop = np.append(l_drop,2020)
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

clf = RandomForestClassifier(n_estimators = 100)


clf.fit(X_train, y_train)

feature_imp = pd.Series(clf.feature_importances_, index = df.columns).sort_values(ascending = False)
print(feature_imp)


y_pred = clf.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))