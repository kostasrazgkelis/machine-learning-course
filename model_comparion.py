from random import randint

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd



def main():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    random_state = randint(0, 1000)

    # Create a decision tree classifier as the base estimator for the Bagging Classifier
    dt = DecisionTreeClassifier(random_state=random_state)

    # Create the Bagging Classifier

    models = {
        ('Bugging', BaggingClassifier(base_estimator=dt, n_estimators=100, random_state=random_state)),
        ('RandomForest', RandomForestClassifier(random_state=random_state))
    }
    results = []
    # Fit the Bagging Classifier to the data
    for name, model in models:

        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Calculate evaluation metrics
        f1 = f1_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)

        results.append({"F1": f1, "Accuracy": acc, "Recall": recall, "Precision": prec})

    # Plotting the result
    metrics = ['F1', 'Accuracy', 'Precision', 'Recall']
    values = results[0].values()
    values_rf = results[1].values()

    fig, ax = plt.subplots()
    index = np.arange(len(metrics))
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, values, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Bagging Classifier')

    rects2 = plt.bar(index + bar_width, values_rf, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Random Forest')

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Metrics comparison between Bagging Classifier and Random Forest')
    plt.xticks(index + bar_width, metrics)
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_comparison.png")

    final_df = pd.DataFrame(results)
    print(final_df)



if __name__ == "__main__":
    main()