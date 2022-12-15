from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from numpy import mean, absolute, sqrt

iris = datasets.load_iris()
X = iris.data[:, :2]  # Load just 2 feature for simplicity and visualization purposes....
y = iris.target

cv = LeaveOneOut()
model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial',  max_iter=3000)

# use LOOCV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# view mean absolute error
MAE = mean(absolute(scores))

# use LOOCV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

# view RMSE
SQRT = sqrt(mean(absolute(scores)))

print(f"The MAE = {MAE} and the SQRT={SQRT}")
