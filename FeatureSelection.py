import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  Normalizer, MinMaxScaler
from sklearn.metrics import  roc_auc_score, accuracy_score, precision_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile

data = pd.read_csv('C:/Users/Onelity-krazgkelis/Desktop/data/HTRU_2.csv', header=None)
data.columns = [f"# {x}" for x in data.columns]

normalizer = Normalizer()

X = data[data.columns[:7]]
y = data[data.columns[8]]

size = X.shape[1]

model = ('LR', LogisticRegression())

selectors = [
    ('SelectKBest', SelectKBest(chi2, k=size)),
    ('SelectPercentile', SelectPercentile(chi2, percentile=20)),
]

# scaling features
scalar = MinMaxScaler()

# fit and transform scalar to train set
X_scaled = scalar.fit_transform(X)

results = []
for name, selector in selectors:
    # Find the best features
    slct = selector.fit_transform(X_scaled, y)
    scores = selector.scores_[selector.get_support()]
    feature_names = X.columns[selector.get_support()]

    scores, feature_names = zip(*sorted(zip(scores, feature_names), reverse=True))

    # Create the bar plot
    plt.bar(feature_names, scores)
    plt.xlabel("Feature")
    plt.ylabel("Score")
    plt.title("Feature Importance Scores")
    plt.show()

    #### WITHOUT PCA ####
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    # transform test set
    X_test = scalar.transform(X_test)

    clf = model[1].fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    json_data = {'ML_model': model[0],
                 'features_model': name,
                 'psa': False,
                 'accuracy': accuracy,
                 'recall': recall,
                 'precision': precision,
                 'f1': f1,
                 'auc': auc,
                 'features': feature_names}

    results.append(json_data)

    #### WITH PCA ####
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    # Create the PCA object
    pca = PCA(n_components=4)

    # Fit and transform the training data
    X_train = pca.fit_transform(X_train)

    # Transform the test data
    X_test = pca.transform(X_test)

    clf = model[1].fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    json_data = {'ML_model': model[0],
                 'features_model': name,
                 'psa': True,
                 'accuracy': accuracy,
                 'recall': recall,
                 'precision': precision,
                 'f1': f1,
                 'auc': auc,
                 'features': feature_names}

    results.append(json_data)

    #### 4 BEST #####
    X = data[[name for name in feature_names[:4]]]

    # fit and transform scalar to train set
    X_scaled = scalar.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    # transform test set
    X_test = scalar.transform(X_test)

    clf = model[1].fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    json_data = {'ML_model': model[0],
                 'features_model': name,
                 'psa': False,
                 'accuracy': accuracy,
                 'recall': recall,
                 'precision': precision,
                 'f1': f1,
                 'auc': auc,
                 'features': feature_names[:4]}
    results.append(json_data)

final_df = pd.DataFrame.from_dict(data=results)
final_df.to_csv("final_results.csv", header=True)