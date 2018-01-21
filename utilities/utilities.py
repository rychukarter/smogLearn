import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_absolute_error


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


def test_one_regression(estimator, X_train, X_test, y_train, y_test, print_results=False):

    lin_reg = estimator
    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)
    y_pred_train = lin_reg.predict(X_train)

    mea_test = mean_absolute_error(y_test, y_pred)
    mea_train = mean_absolute_error(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_pred_train)

    if print_results:
        print("MAE_test:\t", mea_test,
              "\tMAE_train:\t", mea_train,
              "\nR2_test:\t", r2_test,
              "\tR2_train:\t", r2_train,
              "\n-----------------------------------------------------------------")

    return mea_test, mea_train, r2_test, r2_train


def test_regressions(estimators, X_train, X_test, y_train, y_test, data_name='_scores',
                     plot_learning_curves=False, train_sizes=np.linspace(.1, 1.0, 10), save_path=None):

    index = [(x[0] + data_name) for x in estimators]
    results = []
    for name, e in estimators:
        mea_test, mea_train, r2_test, r2_train = test_one_regression(e, X_train, X_test, y_train, y_test)
        results.append([mea_test, mea_train, r2_test, r2_train])
        if plot_learning_curves:
            lc = plot_learning_curve(e, name+data_name, X_train, y_train, train_sizes=train_sizes)
            lc.savefig(save_path+name+data_name+'.png', bbox_inches='tight')
            lc.close()
    return pd.DataFrame(results, columns=['MEA_test', 'MEA_train', 'R2_test', 'R2_train'], index=index)
