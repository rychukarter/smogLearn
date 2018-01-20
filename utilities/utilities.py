import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, explained_variance_score


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def test_regression(estimator, X_train, X_test, y_train, y_test):

    lin_reg = estimator
    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)
    y_pred_train = lin_reg.predict(X_train)

    print("MAE_test:", mean_absolute_error(y_test, y_pred),
          "MAE_train:", mean_absolute_error(y_train, y_pred_train), "\n",
          "EVS_test:", explained_variance_score(y_test, y_pred),
          "EVS_train:", explained_variance_score(y_train, y_pred_train), "\n",
          "R2_test:", r2_score(y_test, y_pred),
          "R2_train:", r2_score(y_train, y_pred_train))