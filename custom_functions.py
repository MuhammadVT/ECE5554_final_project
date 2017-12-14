import pandas as pd
import numpy as np
def fetch_data(camera, image_num):
    """ Fetches the star data from a given image taken from a given camera
    Parameters
    ----------
    camera : str
        The camera name
    image_num: str
        The number of an image from a camera. 
        If set to ".", all the images will be returned
        
    Returns
    -------
    Pandas.DataFrames
        Returns a data fram with NaN entries removed
    """
    # Load all the star data
    data = pd.read_csv('./data/image_database.txt')
    
    # Fetch the data of interest set by "camera" and "image_num"
    regex_txt = "err_.*" + camera + image_num
    df = data.filter(regex=regex_txt)
    
    # Add azimuth feature for pixel location
    regex_txt = "._img_.*" + camera + image_num
    dfn = data.filter(regex=regex_txt)
    for cam in ["mx", "my", "px", "py"]:
        for i in range(5):
            if np.array([cam + str(i) in x for x in df.columns]).any():
                regex_x = "x_img_.*" + cam + str(i)
                regex_y = "y_img_.*" + cam + str(i)
                df.loc[:, "err_azm_" + cam + str(i)] = np.arctan2(dfn.filter(regex=regex_y).as_matrix(), 
                                                            dfn.filter(regex=regex_y).as_matrix())
    
    return df


def prepare_data(df, camera, image_nums=["1", "2"]):
    """ Creates features and split the data into training and testing datasets
    """
    x = []
    y = []
    for l in image_nums:
            regex_txt = "err_.*" + camera + l
            dfn = df.filter(regex=regex_txt)
            dfn.dropna(inplace=True)
            cols = ["err_mag_" + camera + l, "err_ang_"+ camera + l]
            df_tmp = dfn.loc[:, cols]
            y.append(df_tmp.as_matrix())
            cols = ["err_r_" + camera + l, "err_azm_" + camera + l]
            #cols = ["err_r_" + camera + l]
            df_tmp = dfn.loc[:, cols]
            x.append(df_tmp.as_matrix())
            
    y = np.vstack(tuple(y))
    x = np.vstack(tuple(x))
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

