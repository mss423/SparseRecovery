AISTATS 2022 -- Sparse Recovery and Feature Importance for Classification

## Prerequisites:
```
Python3
Matplotlib
Numpy
Scipy
Sklearn
Icecream
Torch
```

## Running the code
To get results, you will need to run the following scripts:

## For recovery of features as a function of number of samples
```
python main.py plot_support
```
## For impurity reductions across left and whole tree
```
python compare_imp.py
```
### Results

After running the above scripts, new plots will be created in the working directory

In the __main__ function, the num_feature parameter corresponds to the number of important features and can be tuned to different levels of sparsity.

In the compare_imp function, the parameters n, p, s, and r correspond to the maximum number of samples, number of features, numbert of important features, and repititions respectively. These can be adjusted for different levels of sparsity.
