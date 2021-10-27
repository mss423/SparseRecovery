import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from icecream import ic


def create_data_binary(m_train, m_test, num_feature, n, seed, path):
    np.random.seed(seed)
    support = np.random.choice(n, num_feature, replace=False)
    theta = np.zeros(n)
    theta[support] = 1

    X_train = np.random.randint(0, 2, (m_train, n))
    X_test = np.random.randint(0, 2, (m_test, n))

    threshold = 0.5
    y_train = (X_train[:, support].mean(axis=1) > threshold).astype(int)
    y_test = (X_test[:, support].mean(axis=1) > threshold).astype(int)

    ans = {
            'theta': theta,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'support': support,
            }

    if path is not None:
        np.savez(path, **ans)
    else:
        return ans

def create_data_linear(m_train, m_test, num_feature, n, sd, seed, verbose, path):
    np.random.seed(seed)
    support = np.random.choice(n, num_feature, replace=False)
    theta = np.zeros(n)
    theta[support] = np.random.randn(num_feature)

    X_train = np.random.randn(m_train, n)
    X_test = np.random.randn(m_test, n)

    y_train = X_train @ theta + np.random.randn(m_train) * sd
    y_test = X_test @ theta + np.random.randn(m_test) * sd

    np.savez(path,
            theta=theta,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            support=support,
            )

def load_data(path, verbose):
    data = np.load(path)
    return data

def eval_support(model_dict, m_train=400, num_feature=9, n=2000, repeat=1):
    ans = dict()
    ic()
    for r in range(repeat):
        data = create_data_binary(m_train, 100, num_feature, n, seed=r, path=None)
        assert data is not None
        X_train = data['X_train']
        y_train = data['y_train']
        support = data['support']
        r = len(support)
        ans.setdefault('Total', []).append(r)
        for name, func in model_dict.items():
            clf = func()
            clf.fit(X_train, y_train)
            if name == 'Lasso':#LogisticRegression':
                f_imp = clf.coef_[0,:]
            else:
                f_imp = clf.feature_importances_
            indexes = np.argsort(-f_imp)
            correct_count = len(set.intersection(set(support), set(indexes[:r])))
            # if name == 'XGB':
            #     print(correct_count)
            ans.setdefault(name, []).append(correct_count)
    for name, val in ans.items():
        ans[name] = np.mean(val)
    ic()
    return ans

def plot_support():
    from functools import partial
    from xgboost import XGBClassifier
    model_dict = {
            #'XGB': partial(XGBClassifier, use_label_encoder=False, eval_metric='mlogloss'),
            #'RandomForestClassifier': RandomForestClassifier,
            'Gradient Boosting': GradientBoostingClassifier,
            'DecisionStump': partial(DecisionTreeClassifier, max_depth=1),
            'Lasso': partial(LogisticRegression, C=2, penalty='l1', solver='liblinear'),
            }
    num_features = [10]#, 20, 30, 50, 100]
    # num_features = [10]

    for p in num_features:
        print()
        # m_trains = np.linspace(50, 500, 10, dtype=int)
        m_trains = np.linspace(50, 1000, 10, dtype=int)
        # m_trains = [50]
        ans_total = dict()
        for m in m_trains:
            ans_now = eval_support(model_dict, m_train=m, num_feature=p, repeat=4)
            for name, val in ans_now.items():
                ans_total.setdefault(name, []).append(val)
        for name, y in ans_total.items():
            plt.plot(m_trains, y, label=name)#, marker='*')
        plt.legend()
        plt.ylabel('Recovery Count')
        plt.xlabel('Training Samples')
        plt.savefig(f'/Users/maxspringer/PycharmProjects/support_p={p}.pdf')
        plt.clf()

def eval_m_n(model_dict, m_train=400, num_feature=9, n=500, repeat=1):
    ans = dict()
    ic()
    for r in range(repeat):
        data = create_data_binary(m_train, 100, num_feature, n, seed=r, path=None)
        assert data is not None
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        for name, func in model_dict.items():
            clf = func()
            clf.fit(X_train, y_train)
            def err_func(y, yp):
                return (y != yp).mean()
            y_test_hat = clf.predict(X_test)
            test_error = err_func(y_test_hat, y_test)
            ans.setdefault(name, []).append(test_error)
    for name, val in ans.items():
        ans[name] = np.mean(val)
    return ans

def plot_m_n():
    from functools import partial
    from xgboost import XGBClassifier
    model_dict = {
            'XGB': partial(XGBClassifier, use_label_encoder=False, eval_metric='mlogloss'),
            'Forest': RandomForestClassifier,
            'LogisticRegression': partial(LogisticRegression, C=2, penalty='l1', solver='liblinear'),
            'svm': svm.LinearSVC,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            }
    num_features = [20, 30, 50, 100]
    # num_features = [10]

    for p in num_features:
        m_trains = np.linspace(50, 500, 10, dtype=int)
        # m_trains = [50]
        ans_total = dict()
        for m in m_trains:
            ans_now = eval_m_n(model_dict, m_train=m, num_feature=p, repeat=10)
            for name, val in ans_now.items():
                ans_total.setdefault(name, []).append(val)
        for name, y in ans_total.items():
            plt.plot(m_trains, y, label=name, marker='*')
        plt.legend()
        plt.ylabel('test_error')
        plt.xlabel('training samples')
        plt.savefig(f'/Users/maxspringer/PycharmProjects/p={p}.pdf')
        plt.clf()

    
def eval_xgboost(data):
    from xgboost import XGBClassifier
    for use_support in [True, False]:
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        support = data['support']
        # clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        # clf = RandomForestClassifier()
        clf = LogisticRegression(C=2, penalty='l1', solver='liblinear')
        # clf = svm.LinearSVC()
        # clf = GradientBoostingClassifier()
        
        if use_support:
            X_train = X_train[:, support]
            X_test = X_test[:, support]
        
        # clf.fit(X_train, y_train, eval_metric='logloss')
        clf.fit(X_train, y_train)
        if hasattr(clf, 'feature_importances_'):
            f_imp = clf.feature_importances_
            indexes = np.argsort(-f_imp)
            r = 9
            print(sorted(indexes[:r]))
            print(sorted(support))
            print('---', len(set.intersection(set(support), set(indexes[:r]))))
        # num = f_imp[indexes][:r]
        # num = f_imp[indexes][11:15]
        # print(num)

        def err_func(y, yp):
            return (y != yp).mean()

        y_train_hat = clf.predict(X_train)
        train_error = err_func(y_train_hat, y_train)

        y_test_hat = clf.predict(X_test)
        test_error = err_func(y_test_hat, y_test)

        print(f"use support: {use_support}")
        print(f"train error: {train_error}")
        print(f"test error: {test_error}")

def eval_models(data):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor
    for par in np.linspace(0, 100, 10):
        name = f"XGboost {par:.2f}"
        regr = XGBRegressor(gamma=par)
        regressors[name] = regr
    err_func = mean_squared_error
    for use_support in [False, True]:
        print('----------------------------')
        xgb_params = {
                'n_estimators': 400,
                'max_depth':1,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 1,
                'reg_alpha': 0.1
                }
        regressors = {
                'Lienar regression': linear_model.LinearRegression(),
                'Lasso': linear_model.Lasso(alpha=0.1),
                'XGboost': XGBRegressor(**xgb_params),
                }
        for name, regr in regressors.items():
            print(f"use_support: {use_support}, Regressor name: {name}")
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            support = data['support']

            if use_support:
                X_train = X_train[:, support]
                X_test = X_test[:, support]
            
            regr.fit(X_train, y_train)

            y_train_hat = regr.predict(X_train)
            train_error = err_func(y_train_hat, y_train)

            y_test_hat = regr.predict(X_test)
            test_error = err_func(y_test_hat, y_test)

            print(f"train error: {train_error}")
            print(f"test error: {test_error}")

def generate(args):
    create_data_binary(
            args.m_train,
            args.m_test,
            args.num_feature,
            args.n,
            #args.sd,
            args.seed,
            #args.verbose,
            args.path
            )

def eval_data(args):
    path = args.path + '.npz'
    data = load_data(path, args.verbose)
#    eval_models(data)
    eval_xgboost(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", type=str)#, default='generate')
    parser.add_argument("--m_train", type=int, default=400)
    parser.add_argument("--m_test", type=int, default=100)
    parser.add_argument("--num_feature", type=int, default=9)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--sd", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path", type=str, default='out')
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    if args.cmd == 'generate':
        generate(args)
    elif args.cmd == 'eval':
        eval_data(args)
    elif args.cmd == 'total':
        generate(args)
        eval_data(args)
    elif args.cmd == 'plot_m_n':
        plot_m_n()
    elif args.cmd == 'plot_support':
        plot_support()
    else:
        assert False


if __name__ == "__main__":
    main()
