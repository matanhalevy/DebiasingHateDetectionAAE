import datetime
from collections import defaultdict
from time import clock
from functools import reduce
import math

from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_sample_weight
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from sklearn.model_selection import learning_curve, GridSearchCV

label = 'IS_DELAYED'
airport = 'BOS'

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, pred_probs):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p, r = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)
    try:
        roc = roc_auc_score(y_true=labels, y_score=pred_probs[:,1])
    except ValueError:
        roc = 0.
    return {
        "acc": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "auc_roc": roc
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, pred_probs):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels, pred_probs)

# Plot yearly distributions of counts
def plot_yearly_distributions(pd_df, title, dataset_name):
    years = sorted(list(pd_df['DEP_YEAR'].unique()))
    num_years = len(years)
    dv_vals = sorted(list(pd_df[label].unique()))
    nrows, ncols = num_years, len(dv_vals)
    print(nrows, ncols)
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * nrows))
        axes = axes.flatten()

        for dv, col, ax in zip(dv_vals * nrows, [col for col in years for _ in range(ncols)], axes):
            temp_df = pd_df[(pd_df[label] == dv) & (pd_df['DEP_YEAR'] == col)]
            temp_df['DEP_MONTH'].hist(by=temp_df[label], xrot=45, ax=ax, bins=12)
            delayed_title = "Delayed " if dv == 1 else "Not Delayed "
            ax.set_title(delayed_title + str(col) + ' Histogram')
            ax.set(xlabel=str(str(col) + ' Distribution'), ylabel=f'count of {str(col)}')

        plt.suptitle(title + ' Distributions', fontsize=25, verticalalignment='baseline')
        # plt.subplots_adjust(left=0.5)
        plt.tight_layout()
        plt.savefig(f"../reports/figures/{dataset_name}_year_histogram_matrix.png")
        plt.show()


# Plot distributions of counts
def plot_distributions(pd_df, title, dataset_name, cols_to_get_distr=None, compare_labels=False):
    if cols_to_get_distr == None:
        cols_to_get_distr = pd_df.columns.values
    num_cols = len(cols_to_get_distr)
    if compare_labels:
        dv_vals = list(pd_df[label].unique())
        dv_vals.sort()
        nrows, ncols = num_cols, len(dv_vals)
        print(nrows, ncols)
    else:
        nrows, ncols = get_middle_factors(num_cols)
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * nrows))
        axes = axes.flatten()

        if compare_labels:
            for dv, col, ax in zip(dv_vals * nrows, [col for col in cols_to_get_distr for _ in range(ncols)], axes):
                temp_df = pd_df[pd_df[label] == dv]
                temp_df[col].hist(by=temp_df[label], xrot=45, ax=ax)
                delayed_title = "Delayed " if dv == 1 else "Not Delayed "
                ax.set_title(delayed_title + col + ' Histogram')
                ax.set(xlabel=str(col + ' Distribution'), ylabel=f'count of {col}')
        else:
            for col, ax in zip(cols_to_get_distr, axes):
                ax.hist(pd_df[col], histtype='bar')

                ax.set_title(col + ' Histogram')
                ax.set(xlabel=str(col + ' Distribution'), ylabel='count of {0}'.format(col))

        plt.suptitle(title + ' Distributions', fontsize=25, verticalalignment='baseline')
        # plt.subplots_adjust(left=0.5)
        plt.tight_layout()
        plt.savefig(f"../reports/figures/{dataset_name}_histogram_matrix.png")
        plt.show()


def plot_distribution(pd_df, title, col_to_get_distr):
    with plt.style.context('Solarize_Light2'):
        plt.figure(figsize=(18, 15))
        plt.hist(pd_df[col_to_get_distr], histtype='bar')
        plt.title(col_to_get_distr + ' distribution')
        plt.xlabel(col_to_get_distr)
        plt.ylabel('count')

        plt.show()


def compare_counts_boxplots(positive_pd, negative_pd, title, dataset_name, positive_label, cols=None):
    # unique_col_values = df_pd[feature_column].unique()
    # print unique_col_values
    if cols is None:
        cols = positive_pd.columns.values.tolist()

    data = []
    labels = []
    for col in cols:
        data.append(positive_pd[col])
        data.append(negative_pd[col])
        labels.append(f'{positive_label}_{col}')
        labels.append(f'not_{positive_label}_{col}')

    with plt.style.context('Solarize_Light2'):
        plt.figure(figsize=(35, 20))
        plt.ylabel('Counts')
        plt.title(title)
        plt.boxplot(data, showfliers=False)
        plt.xticks(np.arange(start=1, stop=len(data) + 1), labels)
        plt.savefig(f"../reports/figures/{dataset_name}_barplots.png")


def create_scatterplot_matrix(pd_df, dataset_name, cols_to_plot=None, label_column=label):
    sns.set(style="ticks")
    if cols_to_plot is None:
        cols_to_plot = pd_df.columns.values
        cols_to_plot = np.delete(cols_to_plot, np.where(cols_to_plot == label_column)).tolist()
    print(cols_to_plot)
    pairplot = sns.pairplot(pd_df, hue=label_column, markers=["o", "+"], vars=cols_to_plot)
    plt.savefig(f"../reports/figures/{dataset_name}_scatterplot_matrix.png")
    return pairplot


def plot_violin_distributions(pd_df, title, dataset_name, label_column, cols_to_plot=None):
    if cols_to_plot is None:
        cols_to_plot = pd_df.columns.values
        cols_to_plot = np.delete(cols_to_plot, np.where(cols_to_plot == label_column)).tolist()
        print(cols_to_plot)
    num_cols = len(cols_to_plot)
    nrows, ncols = get_middle_factors(num_cols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 22))
    axes = axes.flatten()

    for col, ax in zip(cols_to_plot, axes):
        ax.set_title(col + ' violinplot')
        create_violin_plot(pd_df, col, label_column, ax)
        ax.set(ylabel='count of {0}'.format(col))

    plt.suptitle(title, fontsize=16, verticalalignment='baseline')
    # plt.subplots_adjust(left=0.5)

    plt.savefig(f"../reports/figures/{dataset_name}_violinplot_matrix.png")


def create_violin_plot(pd_df, column, label_column, axes):
    sns.set(style="ticks")
    sns.violinplot(data=pd_df, x=label_column, y=column, ax=axes, kind='violin', cut=0)


def plot_distributions_2(pd_df, title, cols_to_get_distr, label):
    dv_vals = list(pd_df[label].unique())
    dv_vals.sort()
    num_axis = (len(cols_to_get_distr), len(dv_vals))
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=num_axis[0], ncols=num_axis[1], figsize=(5 * num_axis[1], 5 * num_axis[0]))
        axes = axes.flatten()
        for dv, col, ax in zip(dv_vals * num_axis[0], [col for col in cols_to_get_distr for _ in range(num_axis[1])],
                               axes):
            temp_df = pd_df[pd_df[label] == dv]
            temp_df[col].hist(by=temp_df[label], xrot=45, ax=ax)

            ax.set_title("Decile Score: " + str(dv) + " (" + col + ")")
            ax.set(xlabel=str(col + ' Distribution'), ylabel=f'count of {col}')

        plt.suptitle(title + ' Distributions', fontsize=30, verticalalignment='top')
        # plt.subplots_adjust(left=0.5)
        plt.savefig(f"./figs/{label}_histogram_matrix.png")
        plt.show()


def get_middle_factors(n: int) -> (int, int):
    step = 2 if n % 2 else 1
    factors = set(reduce(list.__add__, ([i, n // i] for i in range(1, int(np.sqrt(n)) + 1, step) if n % i == 0)))
    factors = np.sort(list(factors))
    print(factors)
    if (len(factors) > 3) & (len(factors) % 2 == 0):
        mid = int(len(factors) / 2)
        return factors[mid - 1], factors[mid]
    elif (len(factors) > 2) & (len(factors) % 2 != 0):
        mid = int(len(factors) / 2)
        return factors[mid], factors[mid]
    elif len(factors) > 1:
        return factors[0], factors[1]
    else:
        return 0


def plot_learning_curve(estimator, title, X, y, algorithm, dataset_name, model_name, y_lim=None, cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5)) (changed to np.linspace(0.1, 1.0, 10))
    """

    plt.figure()
    plt.title(title)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if model_name is 'dt_pruning_1' or model_name is 'boosting_1':
        N = y.shape[0]
        train_sizes = [50, 100] + [int(N * x / 10) for x in range(1, 8)]
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.savefig(f'./figs/learning_curve_{algorithm}_{model_name}_{dataset_name}')
    return plt


def plot_iterative_learning_curve(clfObj, trgX, trgY, tstX, tstY, params, model_name=None, dataset_name=None):
    # also adopted from jontays code
    np.random.seed(42)
    if model_name is None or dataset_name is None:
        raise
    cv = GridSearchCV(clfObj, n_jobs=1, param_grid=params, refit=True, verbose=10, cv=5, scoring='accuracy')
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/ITER_base_{}_{}.csv'.format(model_name, dataset_name), index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:
        d['param_{}'.format(name)].append(value)
        clfObj.set_params(**{name: value})
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(trgX)
        d['train acc'].append(accuracy_score(trgY, pred))
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(tstX)
        d['test acc'].append(accuracy_score(tstY, pred))
        print(value)
    d = pd.DataFrame(d)
    d.to_csv('./output/ITERtestSET_{}_{}.csv'.format(model_name, dataset_name), index=False)
    return d


def make_timing_curve(X_train, y_train, X_test, y_test, clf, model_name, dataset_name, alg):
    # 'adopted' from JonTay's code
    timing_df = defaultdict(dict)
    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        st = clock()
        np.random.seed(42)
        clf.fit(X_train, y_train)
        timing_df['train'][fraction] = clock() - st
        st = clock()
        clf.predict(X_test)
        timing_df['test'][fraction] = clock() - st
        print(model_name, dataset_name, fraction)
    timing_df = pd.DataFrame(timing_df)
    timing_df.to_csv(f'./output/{model_name}_{dataset_name}_timing.csv')

    title = alg + ' ' + dataset_name + ' Timing Curve for Training and Prediction'
    plot_model_timing(title, alg, model_name, dataset_name,
                      timing_df.index.values * 100,
                      pd.DataFrame(timing_df['train'], index=timing_df.index.values),
                      pd.DataFrame(timing_df['test'], index=timing_df.index.values))
    return timing_df


def plot_model_timing(title, algorithm, model_name, dataset_name, data_sizes, fit_scores, predict_scores, ylim=None):
    """
    Generate a simple plot of the given model timing data

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    data_sizes : list, array
        The data sizes

    fit_scores : list, array
        The fit/train times

    predict_scores : list, array
        The predict times

    """
    with plt.style.context('seaborn'):
        plt.close()
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training Data Size (% of total)")
        plt.ylabel("Time (s)")
        fit_scores_mean = np.mean(fit_scores, axis=1)
        fit_scores_std = np.std(fit_scores, axis=1)
        predict_scores_mean = np.mean(predict_scores, axis=1)
        predict_scores_std = np.std(predict_scores, axis=1)

        plt.fill_between(data_sizes, fit_scores_mean - fit_scores_std,
                         fit_scores_mean + fit_scores_std, alpha=0.2)
        plt.fill_between(data_sizes, predict_scores_mean - predict_scores_std,
                         predict_scores_mean + predict_scores_std, alpha=0.2)
        plt.plot(data_sizes, predict_scores_mean, 'o-', linewidth=1, markersize=4,
                 label="Predict time")
        plt.plot(data_sizes, fit_scores_mean, 'o-', linewidth=1, markersize=4,
                 label="Fit time")

        plt.legend(loc="best")
        plt.savefig(f'./figs/timing_curve_{algorithm}_{model_name}_{dataset_name}')
        plt.show()


def _save_cv_results(self):
    # TODO fix this
    regTable = pd.DataFrame(self.dt_model.cv_results_)
    regTable.to_csv(f'./output/cross_validation_{self.model_name}_{self.dataset_name}.csv',
                    index=False)

    results = pd.DataFrame(self.dt_model.cv_results_)
    components_col = 'param___n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    ax = plt.figure()
    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax)
    ax.set_ylabel('Classification accuracy (val)')
    ax.set_xlabel('n_components')
    plt.savefig(f'./figs/cross_validation_{self.model_name}_{self.dataset_name}')
    plt.show()


def save_model(dataset_name, estimator, file_name):
    file_name = dataset_name + '_' + file_name + '_' + str(datetime.datetime.now().date()) + '.pkl'
    joblib.dump(estimator, f'./models/{file_name}')


def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)
