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

    plt.suptitle(suptitle,fontsize=16)
    
    return 