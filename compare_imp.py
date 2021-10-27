import matplotlib.pyplot as plt
import torch
from icecream import ic
from stump import generate_data, eval_method, eval_param
import argparse
import time
import numpy as np
import pandas as pd

def compare_imp():
    ns = range(100,1100,100)
    gini_l = np.array([])
    gini_t = np.array([])
    var_l = np.array([])
    var_t = np.array([])
    for n in ns:
        time_start = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", default=n, type=int)
        parser.add_argument("-p", default=2000, type=int)
        parser.add_argument("-s", default=10, type=int)
        parser.add_argument("-r", default=100, type=int)
        parser.add_argument("--seed", default=0, type=int)
        args = parser.parse_args()
        torch.manual_seed(args.seed)
        kwargs = vars(args)
        for arg, val in kwargs.items():
            print(f"{arg}: {val}")
        del kwargs['seed']
        ans = eval_param(**kwargs)
        for method_name, score in ans.items():
            print(f"{method_name}: {score:.4f}")
        time_end = time.time()
        print(f'Total time: {time_end - time_start}')
        gini_l = np.append(gini_l,ans['gini, left'])
        gini_t = np.append(gini_t,ans['gini, total'])
        var_l = np.append(var_l,ans['var, left'])
        var_t = np.append(var_t,ans['var, total'])

    # Make a data frame
    df = pd.DataFrame({'x': range(100,1100,100), 'Gini Left': gini_l, 'Gini Total': gini_t,
                       'Variance Left': var_l, 'Variance Total': var_t})

    # Change the style of plot
    plt.style.use('seaborn-bright')

    # Create a color palette
    palette = plt.get_cmap('Set1')

    # Plot multiple lines
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

    # Add legend
    plt.legend(loc='center right')

    # Add titles
    # plt.title("Plotting", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Sample Size")
    plt.ylabel("Impurity")

    # Show the graph
    plt.show()

if __name__ == '__main__':
    compare_imp()
