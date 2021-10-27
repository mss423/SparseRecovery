# import numpy as np
import torch
from icecream import ic
CUDA = False

def median_split(X, y):
    n, p = X.shape
    m = n // 2
    permutation = X.argsort(axis=0)
    index_L = permutation[:m, :]
    index_R = permutation[m:, :]
    n_L, n_R = m, n - m
    y_L = y[index_L]
    y_R = y[index_R]
    return y_L, y_R, n_L, n_R

def order_by_variance(X, y, left_only=False):
    y_L, y_R, n_L, n_R = median_split(X, y)
    n_L, n_R = y_L.shape[0], y_R.shape[0]
    if left_only:
        imp = y_L.var(axis=0)
    else:
        imp = y_L.var(axis=0) * n_L + y_R.var(axis=0) * n_R
    ordering = imp.argsort()
    return ordering

def order_by_gini(X, y, apply_threshold=False, left_only=False):
    if apply_threshold:
        y = (y >= 0).float()
    y_L, y_R, n_L, n_R = median_split(X, y)
    p_hat_L = y_L.mean(axis=0)
    p_hat_R = y_R.mean(axis=0)
    gini_L = p_hat_L * (1 - p_hat_L)
    gini_R = p_hat_R * (1 - p_hat_R)
    if left_only:
        imp = gini_L
    else:
        imp = (gini_L * n_L + gini_R * n_R) / (n_L + n_R)
    ordering = imp.argsort()
    return ordering

def order_by_method(X, y, method: str, s : int = None, left_only=False):
    """
    method: 'var' | 'gini'
    """
    if method == 'var':
        ans = order_by_variance(X, y, left_only=left_only)
    elif method == 'gini':
        ans = order_by_gini(X, y, apply_threshold=True, left_only=left_only)
    else:
        raise ValueError()
    if s is None:
        return ans
    else:
        return ans[:s]

def eval_method(X, y, S, method: str,
        metric: str = 'soft', left_only=False):
    """
    metirc: 'soft' or 'hard'
    If soft, the number of correct retreived elemnts is reported as score.
    If hard, 1 if all elements retreived correctly, 0 otherwise.
    output: Single entry
    """
    s = len(S)
    S_hat = order_by_method(X, y, method, s, left_only=left_only)
    # intersection = len(set.intersection(set(S), set(S_hat))) 
    intersection = 2 * s - torch.cat((S, S_hat)).unique().shape[0]
    if metric == 'soft':
        return intersection/s
    elif metric == 'hard':
        return int(intersection == s)
    else:
        raise ValueError()

def generate_data(n: int, p: int, s: int):
    """
    output: X, y, S tuple
    """
    # X = np.random.uniform(-1, 1, (n, p))
    X = torch.rand(n, p) * 2 - 1
    # S = np.random.choice(p, s, False)
    S = torch.randperm(p)[:s]
    # y = X[:, S].sum(axis=1)
    y = X[:, S].sum(dim=1)
    ans = X, y, S
    if CUDA:
      ans = tuple(arr.cuda() for arr in ans)
    return ans

def eval_param(n, p, s, r, metric: str = 'soft',
        return_standard_error=False):
    """
    """
    from scipy.stats import sem
    method_scores = {
            #func, left_only
            ('gini', True): [],
            ('gini', False): [],
            ('var', True): [],
            ('var', False): [],
            }
    for _ in range(r):
        X, y, S = generate_data(n, p, s)
        for (method, left_only), li in method_scores.items():
            score = eval_method(X, y, S, method, metric, left_only=left_only)
            li.append(score)
    ans = dict()
    for (method, left_only), li in method_scores.items():
        li = torch.tensor(li)
        method_name = f"{method}, {'left' if left_only else 'total'}"
        if return_standard_error:
            ans[method_name] = torch.mean(li), sem(li)
        else:
            ans[method_name] = torch.mean(li)
    return ans
