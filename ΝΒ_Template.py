# =============================================================================
# HOMEWORK 5 - BAYESIAN LEARNING
# NAIVE BAYES ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: gliapisa@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for loading data
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'naive_bayes' package, for creating and using Naive Bayes classfier
import numpy as np
from sklearn import datasets, model_selection, metrics, naive_bayes


# We also need to import 'make_pipeline' from the 'pipeline' module.
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# We are working with text, so we need an appropriate package
# that shall vectorize words within our texts.
# 'TfidfVectorizer' from 'feature_extraction.text'.
from sklearn.feature_extraction.text import TfidfVectorizer

# 'matplotlib.pyplot' and 'seaborn' are ncessary as well,
# for plotting the confusion matrix.
import matplotlib.pyplot as plt
import seaborn as sns;


# Load text data.
textData = datasets.fetch_20newsgroups()


# Store features and target variable into 'X' and 'y'.
X = textData.data
y = textData.target
labels = textData.target_names


# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)



# We need to perform a transformation on the model that will later become 
# our Naive Bayes classifier. This transformation is text vectorization,
# using TfidfVectorizer().
# When you want to apply several transformations on a model, and an
# estimator at the end, you can use a 'pipeline'. This allows you to
# define a chain of transformations on your model, like a workflow.
# In this case, we have one transformer that we wish to apply (TfidfVectorizer)
# and an estimator afterwards (Multinomial Naive Bayes classifier).
# =============================================================================


# ADD COMMAND TO MAKE PIPELINE HERE
alpha = 0.1 # This is the smoothing parameter for Laplace/Lidstone smoothing
model = TfidfVectorizer()
x_train = model.fit_transform(x_train)
x_test = model.transform(x_test)


# =============================================================================




    
# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN MODEL HERE
model = MultinomialNB(alpha=7)

model.fit(x_train, y_train)


# =============================================================================




# Ok, now let's predict output for the second subset
# =============================================================================


# ADD COMMAND TO MAKE PREDICTION HERE
y_predicted = model.predict(x_test)
y_predicted = np.asarray(y_predicted).reshape(-1, 1)

# =============================================================================





# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'accuracy_score()', recall_score()', 'precision_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform 
# a type of averaging on the data. Use 'macro' for final results.
# =============================================================================


# ADD COMMANDS TO COMPUTE METRICS HERE (AND PRINT ON CONSOLE)
accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted, average="macro")
precision = metrics.precision_score(y_test, y_predicted, average="macro")
f1 = metrics.f1_score(y_test, y_predicted, average="macro")

print("Accuracy: %f" % accuracy)
print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)


# =============================================================================




# In order to plot the 'confusion_matrix', first grab it from the 'metrics' module
# and then throw it within the 'heatmap' method from the 'seaborn' module.
# =============================================================================

# ADD COMMANDS TO PLOT CONFUSION MATRIX
confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
sns.set(font_scale=0.5)
heat_map = sns.heatmap(confusion_matrix, annot=True, cbar=True, xticklabels=labels, yticklabels=labels, cmap="YlGnBu")

plt.title("Multinomial NB- Confusion matrix (a= 7)")
plt.suptitle("[Precision = %2f , Recall = %2f, F1 = %2f]" % (precision, recall, f1))
plt.show()


# =============================================================================
