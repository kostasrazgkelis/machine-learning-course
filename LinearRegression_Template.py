# =============================================================================
# HOMEWORK 1 - Supervised learning
# LINEAR REGRESSION ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: gliapisa@csd.auth.gr
# =============================================================================



# From 'sklearn' library, we need to import:
# 'datasets', for loading our data
# 'metrics', for measuring scores
# 'linear_model', which includes the LinearRegression() method
# From 'scipy' library, we need to import:
# 'stats', which includes the spearmanr() and pearsonr() methods for computing correlation
# Additionally, we need to import 
# 'pyplot' from package 'matplotlib' for our visualization purposes
# 'numpy', which implementse a wide variety of operations
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, linear_model
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================



# Load diabetes data from 'datasets' class
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# =============================================================================

# Load just 1 feature for simplicity and visualization purposes...
# X: features
# Y: target value (prediction target)
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target
#
# # =============================================================================
#
#
# # Create linear regression model.
# # =============================================================================
#
#
# # ADD COMMAND TO CREATE LINEAR REGRESSION MODEL HERE
regr = linear_model.LinearRegression()


# =============================================================================



# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_size = len(X.data)
y_size = len(y.data)

x_train, x_test = X[:int(np.floor(x_size*3/4))], X[:int(np.ceil(x_size*1/4))]
y_train, y_test = y[:int(np.floor(y_size*3/4))], y[:int(np.ceil(y_size*1/4))]

# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN YOUR MODEL HERE
regr.fit(x_train, y_train)
# =============================================================================




# Ok, now let's predict the output for the test input set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = regr.predict(x_test)

# =============================================================================



# Time to measure scores. We will compare predicted output (resulting from input x_test)
# with the true output (i.e. y_test).
# You can call 'pearsonr()' or 'spearmanr()' methods for computing correlation,
# 'mean_squared_error()' for computing MSE,
# 'r2_score()' for computing r^2 coefficient.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Coefficients: %.2f" % regr.coef_[0])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_predicted))

# =============================================================================




# Plot results in a 2D plot (scatter() plot, line plot())
# =============================================================================

# ADD COMMANDS FOR VISUALIZING DATA (SCATTER PLOT) AND REGRESSION MODEL
plt.scatter(x=x_test[:], y=y_test[:], color="black")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.plot(x_test, y_predicted, color="blue", linewidth=3)

# Display 'ticks' in x-axis and y-axis
plt.xticks()
plt.yticks()

# Show plot
plt.show()

# =============================================================================
