# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:39:13 2017

@author: addiewan
surface fitting code adopted from:
    https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
"""



def err_maps(img=['mx3'],
                fit='linear',
                order=None,
                rows=1520,
                cols=1520,
                plot='no'):
    '''
    Input:
        x_img: numpy array of x-pixel centroid locations
        y_img: numpy array of y-pixel centroid locations
        x_act: numpy array of x-pixel modeled star locations
        y_act: numpy array of y-pixel modeled locations
        fit: 'linear','nearest', 'polyfit'type of surface fitting to use
        order: if polyfit chosen, specify order of polyfit (ie order=3)
        
    Output:
        err_maps: dict containing the following:
              'img': list of images used to create error maps
              'offx_1520': x-pixel offsets for a 1520x1520 grid
              'offy_1520': y-pixel offsets for a 1520x1520 grid
              'err_mag_1520': pixel error magnitude for a 1520x1520 grid
              'offx_1360': x-pixel offsets for a 1360x1360 grid
              'offy_1360': y-pixel offsets for a 1360x1360 grid
              'err_mag_1360': pixel error magnitude for a 1360x1360 grid
              'offx_sci': x-pixel offsets for a 340x170 science binned grid
              'offy_sci': y-pixel offsets for a 340x170 science binned grid
              'err_mag_sci': pixel error magnitude for a 340x170 science binned grid
              
    Example Usage:
        1) Create/plot error map using 'mx3' stars and 2d polyfit order 2
            
            import matplotlib.pyplot as plt
            from err_maps import err_maps
            img=['mx3']
            err_maps = err_maps(img=img,fit='2d polyfit',order=2)
            plt.imshow(err_maps['err_mag_1520'],interpolation=None)
            plt.title('Error map offset magnitude \n'+str(img))
            plt.xlabel('x-pixel coordinates')
            plt.ylabel('y-pixel coordinates')
            plt.colorbar()
            
        2) Create/plot error map using stars from ['mx1','mx2','mx3'] and 
           2d polyfit with order=2
            
            import matplotlib.pyplot as plt
            from err_maps import err_maps
            img=['mx1','mx2','mx3']
            err_maps = err_maps(img=img,fit='2d polyfit',order=2)
            plt.imshow(err_maps['err_mag_1520'],interpolation=None)
            plt.title('Error map offset magnitude \n'+str(img))
            plt.xlabel('x-pixel coordinates')
            plt.ylabel('y-pixel coordinates')
            plt.colorbar()
    '''
    import itertools
    import numpy as np
    from scipy.interpolate import griddata
    from star_utils import img_stars,load_stars,rebin,trim
    import scipy.linalg
    
    data = load_stars()
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
 
    xdiff = x_img-x_act
    ydiff = y_img-y_act
    mag = np.sqrt((x_img-x_act)**2+(y_img-y_act)**2)
    angle = np.arctan2(y_img-759.5,x_img-759.5)
    for i in range(0,angle.size):
        if angle[i] < 0:
            angle[i] = angle[i] + 360*np.pi/180.0
    x = np.linspace(-rows/2.0 + .5,rows/2.0 - .5, rows)
    y = np.linspace(-cols/2.0 + .5,cols/2.0 - .5, cols)
    x,y = np.meshgrid(x,y)
    
    if fit == 'linear':
        xy = np.array([x_img-759.5,y_img-759.5]).transpose()
        zz_x = griddata(xy,xdiff,(x,y),method='linear')
        zz_y = griddata(xy,ydiff,(x,y),method='linear')
        zz_mag = griddata(xy,mag,(x,y),method='linear')
        
    if fit == 'nearest':
        xy = np.array([x_img-759.5,y_img-759.5]).transpose()
        zz_x = griddata(xy,xdiff,(x,y),method='nearest')
        zz_y = griddata(xy,ydiff,(x,y),method='nearest')
        zz_mag = griddata(xy,mag,(x,y),method='nearest')
        
    if fit == 'surface':
       order = 2
       data_xdiff = np.array([x_img,y_img,xdiff]).T
       data_ydiff = np.array([x_img,y_img,ydiff]).T 
       data_mag = np.array([x_img,y_img,mag]).T 
       
       data = data_xdiff/100.0
       rows=15
       cols=15
       x = np.linspace(-rows/2.0 + .5,rows/2.0 - .5, rows)
       y = np.linspace(-cols/2.0 + .5,cols/2.0 - .5, cols)
       X,Y = np.meshgrid(x,y)
       XX = X.flatten()
       YY = Y.flatten()
       # best-fit quadratic curve
       A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
       C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
       # evaluate it on a grid
       zz_x = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
       zz_x = scipy.misc.imresize(zz_x, (1520,1520))
       
       data = data_ydiff/100.0
       rows=15
       cols=15
       x = np.linspace(-rows/2.0 + .5,rows/2.0 - .5, rows)
       y = np.linspace(-cols/2.0 + .5,cols/2.0 - .5, cols)
       X,Y = np.meshgrid(x,y)
       XX = X.flatten()
       YY = Y.flatten()
       # best-fit quadratic curve
       A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
       C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
       
       # evaluate it on a grid
       zz_y = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
       zz_y = scipy.misc.imresize(zz_y, (1520,1520))
       
       data = data_mag/100.0
       rows=15
       cols=15
       x = np.linspace(-rows/2.0 + .5,rows/2.0 - .5, rows)
       y = np.linspace(-cols/2.0 + .5,cols/2.0 - .5, cols)
       X,Y = np.meshgrid(x,y)
       XX = X.flatten()
       YY = Y.flatten()
       # best-fit quadratic curve
       A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
       C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
       
       # evaluate it on a grid
       zz_mag = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
       zz_mag = scipy.misc.imresize(zz_mag, (1520,1520))
        
    if fit == '2d polyfit':
        def polyfit2d(x, y, z, order=order):
            ncols = (order + 1)**2
            G = np.zeros((x.size, ncols))
            ij = itertools.product(range(order+1), range(order+1))
            for k, (i,j) in enumerate(ij):
                G[:,k] = x**i * y**j
            m, _, _, _ = np.linalg.lstsq(G, z)
            return m
            
        def polyval2d(x, y, m):
            order = int(np.sqrt(len(m))) - 1
            ij = itertools.product(range(order+1), range(order+1))
            z = np.zeros_like(x)
            for a, (i,j) in zip(m, ij):
                z += a * x**i * y**j
            return z
        
        # Generate Data...
        x = x_img
        y = y_img
        z_x = xdiff
        z_y = ydiff
        z_mag = mag
    
        # Fit a 3rd order, 2d polynomial
        m_x = polyfit2d(x,y,z_x)
        m_y = polyfit2d(x,y,z_y)
        m_mag = polyfit2d(x,y,z_mag)
    
        # Evaluate it on a grid...
        #rows, cols = 1520, 1520
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), rows), 
                             np.linspace(y.min(), y.max(), rows))
        zz_x = polyval2d(xx, yy, m_x)
        zz_y = polyval2d(xx, yy, m_y)
        zz_mag = polyval2d(xx, yy, m_mag)
        

    offx_1520=zz_x
    offy_1520=zz_y
    err_mag_1520=zz_mag
    
    offx_1360 = trim(zz_x,(1360,1360))
    offy_1360 = trim(zz_y,(1360,1360))
    err_mag_1360 = trim(zz_mag,(1360,1360))
    
    offx_sci=rebin(offx_1360,(340,170))/8.0
    offy_sci=rebin(offy_1360,(340,170))/4.0
    err_mag_sci=np.sqrt(offx_sci**2 + offy_sci**2)
    
    err_maps={'img':img,
              'offx_1520':offx_1520,
              'offy_1520':offy_1520,
              'err_mag_1520':err_mag_1520,
              'offx_1360':offx_1360,
              'offy_1360':offy_1360,
              'err_mag_1360':err_mag_1360,
              'offx_sci':offx_sci,
              'offy_sci':offy_sci,
              'err_mag_sci':err_mag_sci}
    
    return err_maps

def err_compare(img,err_maps,plot_dmap=False):
    '''
    Apply an error map to the centroid coordinates of a given list of images
    and return statistics of the error. Option to plot distortion map.
    
    inputs:
        img: list of image to test error map on ie ['mx2']
        err_map: error maps to apply to image
        dmap: set to True to plot resulting distortion map
        
    Outputs:
        err_stats: statistics on error between modeled star pixel locations and
                   the centroid pixel locations that were 'corrected' by error map
                   
    Example Usage:
        1) create error map for 'mx' camera using images 'mx2' and 'mx3' and then
            apply error map to image 'mx1' and see how well the error is corrected
            
            import pandas as pd
            from err_maps import err_maps,err_compare
            err_maps = err_maps(['mx1','mx2'],fit='2d polyfit',order=2)
            err_stats = err_compare(['mx3'],err_maps,plot_dmap=True)
    '''
    from star_utils import img_stars,load_stars
    from dmap import dmap
    import numpy as np
    import pandas as pd
    
    data=load_stars()
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
    
    x_img_new=[]
    y_img_new=[]
    for i in range(len(x_img)):
        x_img_new.append(x_img[i] - err_maps['offx_1520'][int(x_img[i]),int(y_img[i])])
        y_img_new.append(y_img[i] - err_maps['offy_1520'][int(x_img[i]),int(y_img[i])])
    x_img_new=np.array(x_img_new)
    y_img_new=np.array(y_img_new)
    
    x_err_old = x_img - x_act
    y_err_old = y_img - y_act
    mag_err_old = np.sqrt(x_err_old**2+y_err_old**2)
    
    x_err_new = x_img_new - x_act
    y_err_new = y_img_new - y_act
    mag_err_new = np.sqrt(x_err_new**2+y_err_new**2)
    
    err_stats = {'x_err_old':x_err_old,
                 'y_err_old':y_err_old,
                 'mag_err_old':mag_err_old,
                 'x_err_new':x_err_new,
                 'y_err_new':y_err_new,
                 'mag_err_new':mag_err_new}
    
    if plot_dmap == True:
        data_dmap = {'x_img_'+img[0]:x_img,
                    'y_img_'+img[0]:y_img,
                    'x_act_'+img[0]:x_act,
                    'y_act_'+img[0]:y_act,
                    'x_img_new_'+img[0]:x_img_new,
                    'y_img_new_'+img[0]:y_img_new,
                    'mag_err_old_'+img[0]:mag_err_old,
                    'mag_err_new_'+img[0]:mag_err_new}
        err_data = pd.DataFrame(data_dmap)
        
        dmap(err_data,
             x_act='x_act_',
             y_act='y_act_',
             x_img='x_img_',
             y_img='y_img_',
             err_mag='mag_err_old_',
             title='Original Distortion Map',
             img_list=img)
        dmap(err_data,
             x_act='x_act_',
             y_act='y_act_',
             x_img='x_img_new_',
             y_img='y_img_new_',
             err_mag='mag_err_new_',
             title='Distortion Map after error correction',
             img_list=img)
    
    return err_stats
    
    
    
    
    
    