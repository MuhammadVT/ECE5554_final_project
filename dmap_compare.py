# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:25:11 2017

@author: addiewan
"""
def dmap_compare(d_old_train,
                 d_old_test,
                 d_new_train,
                 d_new_test,
                 suptitle='Distortion Map Comparison',
                 title1='Original Distortion Map',
                 title2='Distortion Map after error correction'):
    '''
    Plot distortion maps before and after error correction
    Input:
        d_old_train: dict of distortion parameters for original training dataset
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
        d_old_test: distortion parameters for original testing dataset
        d_new_train: distortion parameters for training dataset after error correction
        d_new_test: distortion parameters for testing dataset after error correction
        suptitle: Title of figure
        title1: Title of subplot 1: distortion map before error correction
        title2: Title of subplot 2: distortion map after error correction
            
            
    Output:
        Plot of distortion maps before and after error correction
        
    Example Usage:
        1)Plot distortion map using 'mx2' as a training set and 'mx3' as a test set
            from gen_d_params import gen_d_params   
            from err_maps import err_maps
            from dmap_compare import dmap_compare
    
            train_set=['mx2']
            test_set=['mx3']

            d_orig_train = gen_d_params(train_set)
            d_orig_test = gen_d_params(test_set)
            
            fit=['nearest']
            order=[None]
            e_maps_nearest,d_new_train_nearest,d_new_test_nearest = err_maps(d_orig_train,d_orig_test,fit[0],order[0])
            
            dmap_compare(d_orig_train,d_orig_test,d_new_train_nearest,d_new_test_nearest,
            suptitle='Nearest Neighbor')
            
    '''
    
    from dmap import dmap
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    
    dmap_old = dmap(d_old_test,show_plot=False)
    dmap_new = dmap(d_new_test,show_plot=False)

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

    plt.suptitle(suptitle,fontsize=20)
    
    return 