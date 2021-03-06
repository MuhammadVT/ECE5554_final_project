# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:11:50 2017

@author: addiewan
"""

import numpy as np
from glob import glob
import pandas as pd
import os


def load_stars(loc=os.getcwd()+'/data/image_database.txt'):
    '''
    Load the star database (pandas df) from a .txt file
    
    Example Usage:
        1) 
        
        from star_utils import load_stars
        data = load_stars()
    '''
    df = pd.read_csv(loc, index_col=[0])
    return df

def img_list():
    '''
    Return a list of all star images
    
    Exmple Usage:
        1) img_list = img_list()
    '''
    img_list = ['px1',
                'px2',
                'px3',
                'mx1',
                'mx2',
                'mx3',
                'py1',
                'py2',
                'py3',
                'py4',
                'my3',
                'my4']
    return img_list

def img_stars(data,img,col_list=['ra_act','dec_act']):
    '''
    Return info on stars for a list of images in the star database
    
    input: image (ie 'px1' or 'mx3')
    
    Example usage:
        
        1) return list of star info for single star
        from star_utils import img_stars,load_stars
        data = load_stars()
        img = 'px1'
        col_list = ['ra_act',
                    'dec_act',
                    'x_act_'+img,
                    'y_act_'+img,
                    'x_act_model_'+img,
                    'y_act_model_'+img,
                    'x_img_'+img,
                    'y_img_'+img]
        stars = img_stars(data,img,col_list)
        
        2) create list of x_img,y_img,x_act,y_act for multiple images
        
        from star_utils import img_stars,load_stars
        data = load_stars()
        img = ['px1','px2','px3']
        x_act=[]
        y_act=[]
        x_img=[]
        y_img=[]
        for i in range(len(img)):
            col_list = ['x_act_'+img[i],
                        'y_act_'+img[i],
                        'x_img_'+img[i],
                        'y_img_'+img[i]]
            df = img_stars(data,img[i],col_list)
            x_act.append(list(df['x_act_'+img[i]]))
            y_act.append(list(df['y_act_'+img[i]]))
            x_img.append(list(df['x_img_'+img[i]]))
            y_img.append(list(df['y_img_'+img[i]]))
        x_act = np.array([item for sublist in x_act for item in sublist])
        y_act = np.array([item for sublist in y_act for item in sublist])
        x_img = np.array([item for sublist in x_img for item in sublist])
        y_img = np.array([item for sublist in y_img for item in sublist])


                
            
    
        
    '''
    
    img_stars = data.loc[~np.isnan(data['x_act_'+img]),col_list]
    
    return img_stars

def load_hi_res(loc = os.getcwd()+'/data/high_res/',
                filter_bad='yes'):
    '''
    Load all the .fits images from the high resolution datset and create
    
    Input:
        loc = directory location of .fits files
        
        filter_bad = 'yes' (default) to remove all images that are labeled 'bad'
        
    Output:
        img_fits = dict containing all image arrays
        
    Example Usage:
        1) img_fits = load_hi_res()
    '''
    from astropy.io import fits
    
    img_loc = glob(loc+'*.fits')
    
    img_fits={}
    for img in img_loc:
        idl_data = (fits.open(img))[0].data
        img_fits[img.rsplit('\\')[-1][:-5]] = idl_data
    
    #filter out bad images
    for key in list(img_fits.keys()):
        if 'bad' in key:
            del img_fits[key]
            
    #rename to image names
    for key in list(img_fits.keys()):
        img_fits[key[-3:]] = img_fits[key]
        del img_fits[key]
            
    return img_fits


def centroid_star(image,thresh=None,fwhm=None):
    '''
    Given an image, locate centroids of all possible stars in image
    
    Input:
        image = numpy array of image data
        thresh = # of standard deviations above background level that 'peaks' must have to be classified as a star 
        fwhm = typical full-width half-max of stars in image
        
    Output:
        centroids: dict containing
                    x_img: x-coordinates of centroid pixel location
                    y_img: y-coordinates of centroid pixel location
                    sources: table of centroid info
                    
    Example Usage:
        1) Locate all centroids of stars in image 'px3' using default settings
    
            img_fits = load_hi_res()
            centroids = centroid_star(img_fits['px3'])
    '''

    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, iters=5)
    if thresh == None:    
        if fwhm == None:
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    
    if thresh != None: 
        if fwhm == None:
            daofind = DAOStarFinder(fwhm=3.0, threshold=thresh*std)
        if fwhm != None:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std)
    sources = daofind(image - median)    
    x_img = sources[1][:]
    y_img = sources[2][:]
    good=np.argsort(y_img)
    for i in range(x_img.size):
        print(x_img[good[i]],y_img[good[i]])
        
    centroids = {'x_img':x_img,
                 'y_img':y_img,
                 'sources':sources}
    
    return centroids

def compose_rotation(x, y, z):
    '''
    Produce a rotation matrix given three angles for each x,y,z axis
    input: 
        1) x-axis rotation angle (radians)
        2) y-axis rotation angle (radians)
        3) z-axis rotation angle (radians)
    
    output:
        1) rotation matrix
    
    Example Usage:
        1)>>>compose_rotation(45,45,90)
             Out[9]: 
             array([[-0.23538292, -0.7940579 ,  0.56041675],
                    [ 0.46963611,  0.41190357,  0.78088244],
                    [-0.85090352,  0.44699833,  0.27596319]])
    
    '''
    r_x = np.eye(3,3)
    r_y = np.eye(3,3)
    r_z = np.eye(3,3)
    
    r_x[1,1] = np.cos(x)
    r_x[1,2] = -np.sin(x)
    r_x[2,1] = np.sin(x)
    r_x[2,2] = np.cos(x)
    
    r_y[0,0] = np.cos(y)
    r_y[0,2] = np.sin(y)
    r_y[2,0] = -np.sin(y)
    r_y[2,2] = np.cos(y)
    
    r_z[0,0] = np.cos(z)
    r_z[0,1] = -np.sin(z)
    r_z[1,0] = np.sin(z)
    r_z[1,1] = np.cos(z)
    
    #r = r_z*r_y*r_x;
    r = np.dot(np.dot(r_z,r_y),r_x)
    return r

def unit_vector(vector):
    ''' 
    Returns the unit vector of the vector.
    
    Example Usage:
        1) >>>unit_vector([0,23,4])
            out: array([ 0.        ,  0.98521175,  0.17134117])
    '''
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    '''
    Returns the angle in radians between vectors 'v1' and 'v2'::

    Example Usage:
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def load_quats(loc_q_sc = os.getcwd()+'/data/q_sc.sav',
               loc_q_global = os.getcwd()+'/data/q_global.sav'):
    '''
    Load all quaternions from a file location
    
    Input:
        loc_q_sc: location of spacecraft quaternion file q_sc.sav
        loc_q_global: location of global quaternions file q_global.sav
        
    Output:
        quats: dict contatining quaternion objects q_sc an q_global
        
    Example Usage:
        1) quats = load_quats()
    '''
    from pyquaternion import Quaternion
    from scipy.io import readsav
    
    q_sc = readsav(loc_q_sc, python_dict=True)
    for cam in list(q_sc.keys()):
        q_sc[cam] = Quaternion((q_sc[cam][3],q_sc[cam][0],q_sc[cam][1],q_sc[cam][2])) 
        
    q_global = readsav(loc_q_global, python_dict=True)
    for img in list(q_global.keys()):
        q_global[img] = Quaternion((q_global[img][3],q_global[img][0],q_global[img][1],q_global[img][2]))
  
    quats={'q_sc':q_sc,
            'q_global':q_global}   
     
    return quats

def load_cam_angles():
    '''
    Return camera angles (in degrees) for each camera and the associated camera key
        ie x_ang[1] returns angle offset in degrees for camera 'mx'
    
    Example Usage:
        1) cam_angles = load_cam_angles()
    '''
    
    x_ang = np.array((0.42999997,-0.13999999,-19.109999,19.499999))
    y_ang = np.array((38.539998,-39.269998,-0.13999999,-0.45999997))
    z_ang = np.array((0.0,0.0,0.0,0.0))
    cam_list = ['px','mx','py','my']
    
    cam_angles = {'x_ang':x_ang,
                  'y_ang':y_ang,
                  'z_ang':z_ang,
                  'cam_list':cam_list}
    
    return cam_angles

def rebin(array, shape):
    '''
    Rebin an array
    Input:
        array: numpy array
        shape: shape to rebin array to
    Output:
        rebinned array
        
    Example Usage:
        1)  #create a 3x6 array
            array = np.array([[1, 2, 3, 4, 5, 6],
                              [1, 3, 1, 3, 1, 3],
                              [1, 2, 3, 4, 5, 6]])
            
            a)  #bin array along columns
                >>>rebin(array,(3,3))
                Out: array([[ 1.5,  3.5,  5.5],
                            [ 2. ,  2. ,  2. ],
                            [ 1.5,  3.5,  5.5]])
            
            b)  #bin array along rows
                >>>rebin(test,(1,6))
                Out: array([[ 1.0,  2.333,  2.333,  3.667,  3.667,  5.0]])
                
            c)  #bin array along all columns
                >>>rebin(array,(3,1))
                Out: array([[ 3.5],
                            [ 2. ],
                            [ 3.5]])
            
    '''
    sh = shape[0],array.shape[0]//shape[0],shape[1],array.shape[1]//shape[1]
    return array.reshape(sh).mean(-1).mean(1)

def trim(array,shape,off_row=0,off_col=0):
    '''
    Trim an array to a specified size 
    Input:
        array: numpy array
        shape: shape to trim array to
    Output:
        rebinned array
        
    Example Usage:
        1)  #create a 6x6 array
            array = np.array([[1, 2, 3, 4, 5, 6],
                              [6, 5, 4, 3, 2, 1],
                              [8, 9, 1, 2, 3, 4],
                              [5, 6, 7, 8, 9, 1],
                              [2, 3, 4, 5, 6, 7],
                              [1, 2, 3, 4, 5, 6]])
            
            a)  #trim to middle 4x4 area of array
                >>>trim(array,(4,4))
                Out: array([[5, 4, 3, 2],
                            [9, 1, 2, 3],
                            [6, 7, 8, 9],
                            [3, 4, 5, 6]])
            
            b)  #trim to the middle 4x2 area of array
                >>>trim(array,(4,2))
                Out: array([[4, 3],
                            [1, 2],
                            [7, 8],
                            [4, 5]])
            
            c)  #trim to the middle 4x2 area of array and offset col by -1
                >>>trim(array,(4,2),off_col=-1)
                Out: array([[4, 3],
                            [1, 2],
                            [7, 8],
                            [4, 5]])
            
            d)  #trim to the middle 2x2 area of array and offset -1 col, +1 row
                >>>trim(array,(2,2),off_col=-1,off_row=1)
                Out: array([[6, 7],
                            [3, 4]])
    '''
    rows_old=array.shape[0]    
    cols_old=array.shape[1]
    rows_new=shape[0]
    cols_new=shape[1]    
    array = array[int((rows_old-rows_new)/2)+off_row:-int((rows_old-rows_new)/2)+off_row,
                  int((cols_old-cols_new)/2)+off_col:-int((cols_old-cols_new)/2)+off_col]
    return array

def shrink(data, rows, cols):
    return data.reshape(rows, int(data.shape[0]/rows), cols, int(data.shape[1]/cols))

def d2pd(d_param):
    import pandas as pd
    
    data = {k: d_param[k] for k in ('starname','x_img', 'y_img', 'x_act','y_act','mag','xdiff','ydiff')}    
    col_list=[]
    for col in data.columns:
        if 'starname' in col:
            col = col.split(sep='_')[0]
            col_list.append(col)
        else:
            col_list.append(col)
    data.columns=col_list
    df = pd.DataFrame(data)
    
    return df

def d_all2pd(d_params,d_labels,cols=['x_img','starname','y_img', 'x_act','y_act','mag','xdiff','ydiff']):
    import pandas as pd
    #import pdb
    
    data=[]
    for i in range(len(d_params)):
        data.append(pd.DataFrame({k+'_'+d_labels[i]: d_params[i][k] for k in (cols)}))
        data[i]['starname']='nada'
        for j in range(len(data[i].index)):
            if data[i].loc[data[i].index[j],'starname'] != np.nan:
                data[i].loc[data[i].index[j],'starname'] = data[i].loc[data[i].index[j],'starname_'+d_labels[i]]
        #pdb.set_trace()
        
    data = pd.concat(data)   
    data = data.set_index('starname')
    data = data.groupby(['starname']).first().reset_index()
    data = data.set_index('starname')
    return data
            