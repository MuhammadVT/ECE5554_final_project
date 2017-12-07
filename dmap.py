# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:47:29 2017

@author: addiewan
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def dmap(data,
         x_act='x_act_',
         y_act='y_act_',
         x_img='x_img_',
         y_img='y_img_',
         img_list=['px3'],
         save_dmap=os.getcwd()+'/data/'):
    '''
    Generate distortion for the provided columns in the star database
    
    Input:
        data = star database (pandas df)
        x_act = column name of x-pixel modeled actual star location 
        y_act = column name of y-pixel modeled actual star location 
        x_img = column name of x-pixel centroid star location 
        y_img = column name of y-pixel centroid star location 
    
    Output:
        save_dmap = file location of where to save distortion map
        
    Example usage:
        1) Plot dmap for single image
        
        from gen_stars import gen_stars
        data = gen_stars()
        img = ['mx3']
        dmap(data,
             x_act='x_act_',
             y_act='y_act_',
             x_img='x_img_',
             y_img='y_img_',
             img_list=img)
    
        2) Plot dmaps for all images
        
        from gen_stars import gen_stars
        from star_utils import img_list
        data = gen_stars()
        imgs = img_list()
        dmap(data,
             x_act='x_act_',
             y_act='y_act_',
             x_img='x_img_',
             y_img='y_img_',
             img_list=imgs)
    '''
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    #need to move origin of all pixel coordinates to center of 1520x1520 grid
    offset = 759.5
    
    for img in img_list:
#        vectors_act = list(zip((data[x_act+img]-offset).tolist(),(data[y_act+img]-offset).tolist()))
#        vectors_img = list(zip((data[x_img+img]-offset).tolist(),(data[y_img+img]-offset).tolist()))
#        vectors_err = list(zip(((data[x_img+img]-offset)-(data[x_act+img]-offset)).tolist(),((data[y_img+img]-offset)-(data[y_act+img]-offset)).tolist()))
        
        v_err_x = ((data[x_act+img]-offset)-(data[x_img+img]-offset)).tolist()
        v_err_y = ((data[y_act+img]-offset)-(data[y_img+img]-offset)).tolist()
        v_err_loc_x = (data[x_img+img]-offset).tolist()
        v_err_loc_y = (data[y_img+img]-offset).tolist()
        
        #set colors of vectors to represent the relative magnitude of the error
        v_colors = data['err_mag_'+img].tolist()
        for i in range(len(v_colors)):
            if np.isnan(v_colors[i]) == True:
                v_colors[i] = 0.0
        norm = Normalize()
        norm.autoscale(v_colors)
        colormap = cm.jet
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        
        plt.figure()
        plt.quiver(v_err_loc_x,v_err_loc_y,v_err_x,v_err_y, color=colormap(norm(v_colors)))
        plt.title('Distortion Map '+img)
        plt.colorbar(sm)
        
    return

def plot_offsets(data,col1='x_act_px3',col2='x_img_px3,',title=None):
    '''
    Plot the offsets between two columns
    
    Input:
        data: star database (pandas df)
        col1: first column data
        col2: second column of data
        title: optional plot label
        
    Example usage:
        1) plot difference between x_act and x_img
        
        from gen_stars import gen_stars
        from star_utils import img_list
        data = gen_stars()
        imgs = img_list()
        for img in imgs:
            plot_offsets(data,
                         col1='x_act_'+img,
                         col2='x_img_'+img,
                         title=img)
    '''
    plt.figure()
    if title != None:
        plt.title(title)
    plt.plot(data[col1] - data[col2],'o')
    