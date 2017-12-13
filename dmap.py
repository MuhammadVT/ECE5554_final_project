# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:47:29 2017

@author: addiewan
"""
import numpy as np
import matplotlib.pyplot as plt

def dmap(d_params,
         title=None,
         dmap_info=True,
         save_dmap=None,
         show_plot=True,
         figsize=(5,5)):
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
             err_mag='err_mag_',
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
             err_mag='err_mag_',
             img_list=imgs)
    '''
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    #need to move origin of all pixel coordinates to center of 1520x1520 grid
    offset = 759.5
    x_act = d_params['x_act']
    y_act = d_params['y_act']
    x_img = d_params['x_img']
    y_img = d_params['y_img']
    mag   = d_params['mag']
    

    v_err_x = ((x_act-offset)-(x_img-offset)).tolist()
    v_err_y = ((y_act-offset)-(y_img-offset)).tolist()
    v_err_loc_x = (x_img-offset).tolist()
    v_err_loc_y = (y_img-offset).tolist()
    
    #set colors of vectors to represent the relative magnitude of the error
    v_colors = mag.tolist()
    for i in range(len(v_colors)):
        if np.isnan(v_colors[i]) == True:
            v_colors[i] = 0.0
    norm = Normalize()
    norm.autoscale(v_colors)
    colormap = cm.jet
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    if show_plot == True:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.quiver(v_err_loc_x,v_err_loc_y,v_err_x,v_err_y, color=colormap(norm(v_colors)))
        plt.xlim((-759.5,759.5))
        plt.ylim((-759.5,759.5))
        if title != None:
            plt.title(title)
        else:
            plt.title('Distortion Map')
        plt.colorbar(sm)
#        if save_dmap != None:
#            plt.savefig(os.getcwd()+'/data/dmap_figures/'+img+'.png')
        
    if dmap_info==True:
        return {'v_err_loc_x':v_err_loc_x,
                'v_err_loc_y':v_err_loc_y,
                'v_err_x':v_err_x,
                'v_err_y':v_err_y,
                'v_colors':v_colors,
                'norm':norm,
                'sm':sm}
    else:
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
    