# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:11:00 2017

@author: addiewan
"""
def gen_d_params(img,rows=1520,cols=1520):
    '''
    Load distortion maps paramaters into dict
    Input:
        img: list of images used to produce above data ie (['mx1','mx2'])
        rows: number of rows in image data (default =1520)   
        cols: number of cols in image data (default =1520)
    Output:
        d_orig: dict of distortion the following distortion parameters
            x_img: numpy array of x-pixel centroid locations
            y_img: numpy array of y-pixel centroid locations
            x_act: numpy array of x-pixel modeled star locations
            y_act: numpy array of y-pixel modeled locations
            xdiff: numpy array of x_img-x_act
            ydiff: numpy array of y_img-y_act
            mag: numpy array of magnitude of error (np.sqrt(xdiff**2+ydiff**2))
            img: list of images used to produce above data ie (['mx1','mx2'])
            rows: number of rows (y-pixel coordinates) in images (default = 1520)
            cols: number of cols (x-pixel coordinates) in images (default = 1520)
              
    Example Usage:
        1) Create/plot error maps using stars from 'mx3' and a single linear interpolation
            from gen_d_params import gen_d_params
            img_list=['mx1','mx2']   
            d_params_orig = gen_d_params(img_list)
            
    '''
    import numpy as np
    from star_utils import load_stars,img_stars
    data = load_stars()
    x_act=[]
    y_act=[]
    x_img=[]
    y_img=[]
    starname=[]
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
        starname.append(list(df.index))
    x_act = np.array([item for sublist in x_act for item in sublist])
    y_act = np.array([item for sublist in y_act for item in sublist])
    x_img = np.array([item for sublist in x_img for item in sublist])
    y_img = np.array([item for sublist in y_img for item in sublist])
     
    xdiff = x_img-x_act
    ydiff = y_img-y_act
    mag = np.sqrt((x_img-x_act)**2+(y_img-y_act)**2)     
    
    d_params = {'starname':starname[0],
                'x_img':x_img,
                'x_act':x_act,
                'xdiff':xdiff,
                'y_img':y_img,
                'y_act':y_act,
                'ydiff':ydiff,
                'mag':mag,
                'img':img,
                'rows':rows,
                'cols':cols
                }
    
    return d_params

