# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:30:31 2017

@author: addiewan
"""

def plot_error_maps(d_orig_train,
                    e_maps,
                    show_xdiff=True,
                    show_ydiff=True,
                    show_mag=False,
                    cmap='jet'):
    
    import matplotlib.pyplot as plt
        
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(14, 5), facecolor='w', edgecolor='k')
    fig.set_tight_layout(False)
    img=d_orig_train['img']
    
    ax1=plt.subplot(121)
    plt.imshow(e_maps['offx_1520'][0],interpolation=None,cmap=cmap)
    plt.title('Error map offx \n')
    plt.xlabel('x-pixel coordinates')
    plt.ylabel('y-pixel coordinates')
    plt.colorbar()

    ax2=plt.subplot(122)
    plt.imshow(e_maps['offy_1520'][0],interpolation=None,cmap=cmap)
    plt.title('Error map offy \n'+str(img))
    plt.xlabel('x-pixel coordinates')
    plt.ylabel('y-pixel coordinates')
    plt.colorbar()
        
    if show_mag == True:  
        plt.figure()
        ax3=plt.subplot(121)
        plt.imshow(e_maps['offy_1520'][0],interpolation=None,cmap=cmap)
        plt.title('Error map offy \n'+str(img))
        plt.xlabel('x-pixel coordinates')
        plt.ylabel('y-pixel coordinates')
        plt.colorbar()
    
    return