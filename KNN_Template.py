# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email: gliapisa@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score
random.seed = 42
np.random.seed(666)


def embarked(value) -> int:
    if value == 'S':
        return 1
    if value == 'C':
        return 2
    if value == ('Q' or ' '):
        return 3

# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
titanic = pd.read_csv('titanic.csv')
titanic.drop(['Sex', 'Age', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

titanic['Embarked'] = titanic['Embarked'].apply(lambda x: embarked(x))
titanic['Embarked'] = titanic['Embarked'].fillna(3).astype('int')
titanic['Fare'] = titanic['Fare'].fillna(3).astype('int')

X = titanic.iloc[:, :-1].values
y = titanic.iloc[:, 4].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
#imputer =



# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================
f1_no_impute = []
weights = 'distance'
metrics = 'minkowski'
p = 1
for i in range(1, 200):
    neigh = KNeighborsClassifier(n_neighbors=i, weights=weights, metric=metrics, p=p)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    # Plot the F1 performance results for any combination οf parameter values of your choice.
    # If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
    # =============================================================================

    f1 = np.mean(y_pred != y_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=1)

    f1_no_impute.append({'K': i, 'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec})

df = pd.DataFrame(f1_no_impute)

plt.title(f'k-Nearest Neighbors (Weights = {weights}, Metric = {metrics}, p = {p})')
plt.plot(df['f1'].tolist(), label='without impute')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.show()


pos = df['f1'].idxmax()

print(df.iloc[pos])

