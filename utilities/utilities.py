import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_r_score(r_squared):
    """Calculates correlation coefficient (R score) from determination coefficient (R squared score).

    :param r_squared: determination coefficient
    :return: correlation coefficient
    """
    return np.sqrt(r_squared)


def calculate_rmse_score(mse):
    """Calculates root mean squared error from mean square error.

    :param mse: mean square error
    :return: root mean square error
    """
    return np.sqrt(mse)


def calculate_mape_score(y_test, y_pred):
    """Calculates mean absolute percentage error.

    :param y_test: array of target values.
    :param y_pred: array of predicted values
    :return: MAPE score
    """
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Plots learning curves.

    Function from sci-kit learn examples.

    :param estimator: estimator object under test
    :param title: title to show
    :param X: input features
    :param y: output targets
    :param ylim: y_axis scale
    :param cv: cross-validation object
    :param n_jobs: number of parallel jobs
    :param train_sizes: sizes for each point on learning curves
    :return: plotted figure
    """

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


def plot_error_histogram(y_test, y_pred, title="Prediction error histogram"):
    """Plots error histogram.

    Meant for MAE.

    :param y_test: array of target values
    :param y_pred: array of predicted values
    :param title: title to put on histogram
    :return: plotted figure
    """
    error = y_test - y_pred
    n, bins, patches = plt.hist(error, 50, facecolor='g')
    plt.xlabel('Prediction error')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)
    return plt


def test_one_regression(estimator, X_train, X_test, y_train, y_test, print_results=False):
    """Measures most available scores for one estimator that has not been fit.

    :param estimator: estimator object under test
    :param X_train: train part of input data
    :param X_test: test part of input data
    :param y_train: train part of targets
    :param y_test: test part of targets
    :param print_results: if results shall be printed to console
    :return: array of calculated metrics
    """

    lin_reg = estimator
    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)
    y_pred_train = lin_reg.predict(X_train)

    mea_test = mean_absolute_error(y_test, y_pred)
    mea_train = mean_absolute_error(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, y_pred_train)

    if print_results:
        print("MAE_test:\t", mea_test,
              "\tMAE_train:\t", mea_train,
              "\nR2_test:\t", r2_test,
              "\tR2_train:\t", r2_train,
              "\n-----------------------------------------------------------------")
        plot_error_histogram(y_test, y_pred).show()

    return y_pred, y_pred_train, [mea_test, mea_train, r2_test, r2_train, mse_test, mse_train,
                                  calculate_r_score(r2_test), calculate_r_score(r2_train),
                                  calculate_rmse_score(mse_test), calculate_rmse_score(mse_train),
                                  calculate_mape_score(y_test, y_pred), calculate_mape_score(y_train, y_pred_train)]


def test_regressions(estimators, X_train, X_test, y_train, y_test, data_name='_scores', save_path=None,
                     plot_learning_curves=False, train_sizes=np.linspace(.1, 1.0, 10),
                     plot_histogram=False):
    """Tests multiple estimators once.

    :param estimators: estimator objects under test
    :param X_train: train part of input data
    :param X_test: test part of input data
    :param y_train: train part of targets
    :param y_test: test part of targets
    :param data_name: string to add to each saved output file
    :param save_path: directory to save results - if none results aren't saved
    :param plot_learning_curves: True if user wants to plot learning curves
    :param train_sizes: point to plot on learning curve
    :param plot_histogram: should histogram of MAE be plotted
    :return: Pandas Dataframe with results
    """

    index = [(x[0] + data_name) for x in estimators]
    results = []
    for name, e in estimators:
        y_pred, y_pred_train, scores = test_one_regression(e, X_train, X_test, y_train, y_test)
        results.append(scores)
        if plot_learning_curves:
            lc = plot_learning_curve(e, name+data_name, X_train, y_train, train_sizes=train_sizes)
            if save_path:
                lc.savefig(save_path+name+data_name+'_lc.png', bbox_inches='tight')
            lc.close()
        if plot_histogram:
            hist_test = plot_error_histogram(y_test, y_pred, name+data_name+"_test")
            if save_path:
                hist_test.savefig(save_path+name+data_name+'_hist_test.png', bbox_inches='tight')

            hist_train = plot_error_histogram(y_train, y_pred_train, name+data_name+"_train")
            if save_path:
                hist_train.savefig(save_path+name+data_name+'_hist_train.png', bbox_inches='tight')

            hist_test.close()
            hist_train.close()

    return pd.DataFrame(results, columns=['MEA_test', 'MEA_train', 'R2_test', 'R2_train', 'MSE_test', 'MSE_train',
                                          'R_test', 'R_train', 'RMSE_test', 'RMSE_train', 'MAPE_test', 'MAPE_train'],
                        index=index)


def test_regressions_n(estimators, X, y, n=5, data_name='_scores', save_path=None,
                       plot_learning_curves=False, train_sizes=np.linspace(.1, 1.0, 10),
                       plot_histogram=False):
    """Tests multiple estimators n times and returns means as results

    :param estimators: estimator objects under test
    :param X: input data (features)
    :param y: targets
    :param n: how many iterations to average
    :param data_name: string to add to each saved output file
    :param save_path: directory to save results - if none results aren't saved
    :param plot_learning_curves: True if user wants to plot learning curves
    :param train_sizes: point to plot on learning curve
    :param plot_histogram: should histogram of MAE be plotted
    :return: Pandas Dataframe with results
    """

    index = [(x[0] + data_name) for x in estimators]
    results = []

    for name, e in estimators:
        print(e)
        scores = []
        plt_hist = plot_histogram
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True)
            y_pred, y_pred_train, score = test_one_regression(e, X_train, X_test, y_train, y_test)
            scores.append(score)

            if plt_hist:
                hist_test = plot_error_histogram(y_test, y_pred, name + data_name + "_test")
                if save_path:
                    hist_test.savefig(save_path + name + data_name + '_hist_test.png', bbox_inches='tight')

                hist_train = plot_error_histogram(y_train, y_pred_train, name + data_name + "_train")
                if save_path:
                    hist_train.savefig(save_path + name + data_name + '_hist_train.png', bbox_inches='tight')
                plt_hist = False
                hist_test.close()
                hist_train.close()

        mean_scores = np.mean(scores, axis=0)
        print(mean_scores)
        results.append(mean_scores)

        if plot_learning_curves:
            lc = plot_learning_curve(e, name + data_name, X, y, train_sizes=train_sizes)
            if save_path:
                lc.savefig(save_path + name + data_name + '_lc.png', bbox_inches='tight')
            lc.close()

    return pd.DataFrame(results, columns=['MEA_test', 'MEA_train', 'R2_test', 'R2_train', 'MSE_test', 'MSE_train',
                                          'R_test', 'R_train', 'RMSE_test', 'RMSE_train', 'MAPE_test', 'MAPE_train'],
                        index=index)
