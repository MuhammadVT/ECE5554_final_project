import pandas as pd
import numpy as np
def prepare_data(camera, image_nums):
    """ Prepares the data for creating ML models  
    Parameters
    ----------
    camera : str
        The camera name
    image_nums: list
        The labels of images from a camera. 
        
    Returns
    -------
    numpy arrays
        Returns the perdictors and targets seperately
    """
    from gen_d_params import gen_d_params

    # Fetch the data of interest set by "camera" and "image_nums"
    imgs = [camera + x for x in image_nums]
    d_params = gen_d_params(imgs)
    
    # Add azimuth feature for pixel location
    d_params["err_ang"] = np.rad2deg(np.arctan2(d_params["ydiff"], d_params["xdiff"]))
    d_params["r_img"] = np.sqrt(np.power(d_params["x_img"], 2) + np.power(d_params["y_img"], 2)) 
    d_params["err_mag"] = d_params.pop("mag")
    d_params["azm_img"] = np.rad2deg(np.arctan2(d_params["y_img"], d_params["x_img"]))
            
    x1 = np.reshape(d_params["r_img"], (len(d_params["r_img"]), 1))
    x2 = np.reshape(d_params["azm_img"], (len(d_params["azm_img"]), 1))
    x = np.hstack((x1, x2))
    y1 = np.reshape(d_params["err_mag"], (len(d_params["err_mag"]), 1))
    y2 = np.reshape(d_params["err_ang"], (len(d_params["err_ang"]), 1))
    y = np.hstack((y1, y2))

    return x, y

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    """
    Parameters
    ----------
    learner :  
        the learning algorithm to be trained and predicted on
    sample_size : int
        the size of samples (number) to be drawn from training set
    X_train : Pandas.DataFrame 
        predictors in training set
    y_train : Pandas.DataFrame
        response in training set
    X_test : Pandas.DataFrame
        predictors in testing set
    y_test : Pandas.DataFrame
        response in testing set
        
    Returns
    -------
    dict
        a dictionalry object that holdes various results related to model eveluation 
    """
    
    from time import time
    # use mean absolute error as model eveluation metrix
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    results = {}
    
    # fit the learner to the training data using slicing with 'sample_size'
    X_train_sample = X_train[:sample_size, :]
    y_train_sample = y_train[:sample_size, :]
    start = time() # Get start time
    learner = learner.fit(X_train_sample, y_train_sample)
    end = time() # Get end time
    
    # calculate the training time
    results['train_time'] = [end - start] * y_train.shape[1]
        
    # get the predictions on the test set,
    # then get predictions on the training set
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train_sample)
    end = time() # Get end time
    
    # store the predictions
    results['yhat_train'] = predictions_train
    results['yhat_test'] = predictions_test
    
    # calculate the total prediction time
    results['pred_time'] = [end-start] * y_train.shape[1]
            
    # compute mean_absolute_error on the training and testing sets
    #results['mean_absolute_error_train'] = mean_absolute_error(predictions_train, y_train_sample)
    #results['mean_absolute_error_test'] = mean_absolute_error(predictions_test, y_test)
    results['mean_absolute_error_train'] = np.mean(np.abs(y_train_sample - np.mean(predictions_train, axis=0)), axis=0)
    results['mean_absolute_error_test'] = np.mean(np.abs(y_test - np.mean(predictions_test, axis=0)), axis=0)

    # compute mean_squared_error on the training and testing sets
    #results['mean_squared_error_train'] = mean_squared_error(predictions_train, y_train_sample)
    #results['mean_squared_error_test'] = mean_squared_error(predictions_test, y_test)
    results['mean_squared_error_train'] = np.square(np.abs(y_train_sample - np.mean(predictions_train, axis=0))).mean(axis=0)
    results['mean_squared_error_test'] = np.square(np.abs(y_test - np.mean(predictions_test, axis=0))).mean(axis=0)   
    # some info messages
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # return the results
    return results

def pol2cart(phi, rho):
    import numpy as np
    x = np.multiply(rho, np.cos(phi)) 
    y = np.multiply(rho, np.sin(phi))

    return x, y

def dmap_compare_v2(d_old, d_new,
		    suptitle='Distortion Map Comparison',
		    title1='Original Distortion Map',
		    title2='Distortion Map after error correction'):

    from dmap import dmap
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    dmap_old = dmap(d_old,show_plot=False)
    dmap_new = dmap(d_new,show_plot=False)

    norm = Normalize()
    norm.autoscale(dmap_old['v_colors'])
    colormap = cm.jet
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(14, 5), facecolor='w', edgecolor='k')
    fig.set_tight_layout(False)

    ax1=plt.subplot(121)
    dm=dmap_old
    plt.title(title1)
    plt.quiver(dm['v_err_loc_x'],dm['v_err_loc_y'],
              dm['v_err_x'],dm['v_err_y'],
              color=colormap(norm(dm['v_colors'])))
    fig.colorbar(sm)

    ax2=plt.subplot(122)
    dm=dmap_new
    plt.title(title2)
    plt.quiver(dm['v_err_loc_x'],dm['v_err_loc_y'],
              dm['v_err_x'],dm['v_err_y'],
              color=colormap(norm(dm['v_colors'])))
    fig.colorbar(sm)

    plt.suptitle(suptitle,fontsize=16)

    return



