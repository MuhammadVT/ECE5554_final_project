"""
Modified based on visuals.py availabel in the following link
https://github.com/udacity/machine-learning/tree/master/projects/finding_donors

Author: Maimaitirebike (Muhammad Rafiq) Maimaiti 

"""

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time


def evaluate(results, baseline_error_dict, params="magnitude"):
    """
    Visualization code to display results of various learners.
	
	Parameters
	----------
	learners : sklearn model object
		 a list of supervised learners
	results : 
		a list of dictionaries of the statistic results from 'train_predict()'
	mean_absolute_error : float
	mean_squared_error : float
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (15,9))

    # Constants
    bar_width = 0.3
    #colors = ['#A00000', '#00A000']
    colors = ['#A00000','#00A0A0','#00A000']

    if params == "magnitude":
        idx = 0
    if params == "angle":
        idx = 1

    
    # Super loop to plot six panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'mean_absolute_error_train',
                                    'mean_squared_error_train',
                                    'pred_time', 'mean_absolute_error_test',
                                    'mean_squared_error_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j/3, j%3].bar(i+k*bar_width, results[learner][i][metric][idx],
                                 width = bar_width, color = colors[k])
                ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j/3, j%3].set_xticklabels(["30%", "50%", "100%"])
                ax[j/3, j%3].set_xlabel("Training Set Size")
                ax[j/3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Mean Absolute Error")
    ax[0, 2].set_ylabel("Mean Squared Error")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Mean Absolute Error")
    ax[1, 2].set_ylabel("Mean Squared Error")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("MAE  on Training Subset")
    ax[0, 2].set_title("MSE on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("MAE on Testing Set")
    ax[1, 2].set_title("MSE on Testing Set")
    
    # Add horizontal lines for baseline model 
    ax[0, 1].axhline(y = baseline_error_dict["baseline_prediction_MAE_train"],
                     xmin = -0.1, xmax = 3.0,
                     linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = baseline_error_dict["baseline_prediction_MAE_test"],
                     xmin = -0.1, xmax = 3.0,
                     linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = baseline_error_dict["baseline_prediction_MSE_train"],
                     xmin = -0.1, xmax = 3.0,
                     linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = baseline_error_dict["baseline_prediction_MSE_test"],
                     xmin = -0.1, xmax = 3.0,
                     linewidth = 1, color = 'k', linestyle = 'dashed')

#    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0,
#                     linewidth = 1, color = 'k', linestyle = 'dashed')
#    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1,
#                     color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models",
                fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show() 
