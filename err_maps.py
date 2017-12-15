# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:39:13 2017

@author: addiewan
surface fitting code adopted from:
    https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
"""


def err_maps(d_train,d_test,fit_type,order):
    '''
    Input:
        d_train: dict of distortion parameters for training dataset
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
        d_test: dict of distortion parameters for testing dataset (same as above)
        
        fit_type: 'linear','nearest', or '2d polyfit' type of surface fitting to use
        order: if polyfit chosen, specify order of polyfit (ie order=3)
        
     return e_maps,d_new_test,d_new_train   
    Output:
        d_new_train: dict of distortion parameters for training dataset after error correction
        d_new_test: dict of distortion parameters for testing dataset after error correction
        e_maps: list of dicts with the following
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
        1) Create/plot error maps using stars from ['mx1','mx2'] and a single linear interpolation
            
            from d_params import d_params
            from err_maps_R1 import err_maps
            d_params_orig = d_params(['mx1','mx2'])
            fit=['linear']
            order=[None]
            
            err_maps,d_params_new = err_maps(d_params,fit,order)
        
    '''
    import itertools
    import numpy as np
    from scipy.interpolate import griddata
    from star_utils import rebin,trim
    import scipy.linalg
    import skimage
    from skimage.transform import rescale, resize, downscale_local_mean
    import pdb

    
    x = np.linspace(-d_train['rows']/2.0 + .5,d_train['rows']/2.0 - .5, d_train['rows'])
    y = np.linspace(-d_train['cols']/2.0 + .5,d_train['cols']/2.0 - .5, d_train['cols'])
    x,y = np.meshgrid(x,y)
    d_new_train=[]
    d_new_test=[]
    e_maps={'offx_1520':[],
          'offy_1520':[],
          'err_mag_1520':[],
          'offx_1360':[],
          'offy_1360':[],
          'err_mag_1360':[],
          'offx_sci':[],
          'offy_sci':[],
          'err_mag_sci':[]}
    
    fit_type=[fit_type]
    order = [order]
    
    #Can remove this for loop in the future
    for i in range(len(fit_type)):
        #Discard all NAN values in arrays and store in temporary arrays
        x_img_=d_train['x_img'][~np.isnan(d_train['x_img'])]
        y_img_=d_train['y_img'][~np.isnan(d_train['y_img'])]
        xdiff_=d_train['xdiff'][~np.isnan(d_train['xdiff'])]
        ydiff_=d_train['ydiff'][~np.isnan(d_train['ydiff'])]
        mag_=d_train['mag'][~np.isnan(d_train['mag'])]
        if fit_type[i] == 'linear':
            xy = np.array([x_img_-759.5,y_img_-759.5]).transpose()
            zz_x = griddata(xy,xdiff_,(x,-y),method='linear')
            zz_y = griddata(xy,ydiff_,(x,-y),method='linear')
            zz_mag = griddata(xy,mag_,(x,-y),method='linear')
            
        if fit_type[i] == 'nearest':
            xy = np.array([x_img_-759.5,y_img_-759.5]).transpose()
            zz_x = griddata(xy,xdiff_,(x,-y),method='nearest')
            zz_y = griddata(xy,ydiff_,(x,-y),method='nearest')
            zz_mag = griddata(xy,mag_,(x,-y),method='nearest')
            
        if fit_type[i] == 'surface':
           order = 2
           data_xdiff = np.array([x_img_,y_img_,xdiff_]).T
           data_ydiff = np.array([x_img_,y_img_,ydiff_]).T 
           data_mag = np.array([x_img_,y_img_,mag_]).T 
           
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
           zz_x = skimage.transform.resize(zz_x, (1520,1520))
           
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
            
        if fit_type[i] == '2d polyfit':
            def polyfit2d(x, y, z, order=order[i]):
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
            x = x_img_
            y = y_img_
            z_x = xdiff_
            z_y = ydiff_
            z_mag = mag_
        
            # Fit a 3rd order, 2d polynomial
            m_x = polyfit2d(x,-y,z_x)
            m_y = polyfit2d(x,-y,z_y)
            m_mag = polyfit2d(x,-y,z_mag)
        
            # Evaluate it on a grid...
            #rows, cols = 1520, 1520
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), d_train['cols']), 
                                 np.linspace(y.min(), y.max(), d_train['rows']))
            zz_x = polyval2d(xx, yy, m_x)
            zz_y = polyval2d(xx, yy, m_y)
            zz_mag = polyval2d(xx, yy, m_mag)
        

        #Generate Error map data
        e_maps['img']=d_train['img']
        
        e_maps['offx_1520'].append(zz_x)
        e_maps['offy_1520'].append(zz_y)
        e_maps['err_mag_1520'].append(zz_mag)
        
        e_maps['offx_1360'].append(trim(zz_x,(1360,1360)))
        e_maps['offy_1360'].append(trim(zz_y,(1360,1360)))
        e_maps['err_mag_1360'].append(trim(zz_mag,(1360,1360)))
        
        e_maps['offx_sci'].append(rebin(e_maps['offx_1360'][i],(340,170))/8.0)
        e_maps['offy_sci'].append(rebin(e_maps['offy_1360'][i],(340,170))/4.0)
        e_maps['err_mag_sci'].append(np.sqrt(e_maps['offx_sci'][i]**2 + e_maps['offy_sci'][i]**2))
        
        #Apply error maps created with training data to TRAINING set
        x_img_new_train=[]
        y_img_new_train=[]
        #pdb.set_trace()
        for j in range(len(d_train['x_img'])):
            if np.isnan(d_train['mag'][j]):
                x_img_new_train.append(np.nan)
                y_img_new_train.append(np.nan)
            if np.isnan(d_train['mag'][j]) == False:
                x_img_new_train.append(d_train['x_img'][j] - e_maps['offx_1520'][i][int(d_train['x_img'][j]),int(d_train['y_img'][j])])
                y_img_new_train.append(d_train['y_img'][j] - e_maps['offy_1520'][i][int(d_train['x_img'][j]),int(d_train['y_img'][j])])
        x_img_new_train=np.array(x_img_new_train)
        y_img_new_train=np.array(y_img_new_train)
        
        xdiff_new_train = x_img_new_train - d_train['x_act']
        ydiff_new_train = y_img_new_train - d_train['y_act']
        mag_new_train = np.sqrt(xdiff_new_train**2+ydiff_new_train**2)
        
        d_new_train = {'starname':d_train['starname'],
                       'x_img':x_img_new_train,
                       'x_act':d_train['x_act'],
                       'xdiff':xdiff_new_train,
                       'y_img':y_img_new_train,
                       'y_act':d_train['y_act'],
                       'ydiff':ydiff_new_train,
                       'mag':mag_new_train,
                       'img':d_train['img'],
                       'rows':d_train['rows'],
                       'cols':d_train['cols']
                       }
        
        
        
        #Apply error maps created with training data to TESTING set
        x_img_new=[]
        y_img_new=[]
        #pdb.set_trace()
        for j in range(len(d_test['x_img'])):
            
            
            if np.isnan(d_test['mag'][j]):
                x_img_new.append(np.nan)
                y_img_new.append(np.nan)
            if np.isnan(d_test['mag'][j]) == False:
                x_img_new.append(d_test['x_img'][j] - e_maps['offx_1520'][i][int(d_test['x_img'][j]),int(d_test['y_img'][j])])
                y_img_new.append(d_test['y_img'][j] - e_maps['offy_1520'][i][int(d_test['x_img'][j]),int(d_test['y_img'][j])])
        x_img_new=np.array(x_img_new)
        y_img_new=np.array(y_img_new)
    
        
        xdiff_new = x_img_new - d_test['x_act']
        ydiff_new = y_img_new - d_test['y_act']
        mag_new = np.sqrt(xdiff_new**2+ydiff_new**2)
        
        d_new_test = {'starname':d_test['starname'],
                      'x_img':x_img_new,
                      'x_act':d_test['x_act'],
                      'xdiff':xdiff_new,
                      'y_img':y_img_new,
                      'y_act':d_test['y_act'],
                      'ydiff':ydiff_new,
                      'mag':mag_new,
                      'img':d_test['img'],
                      'rows':d_test['rows'],
                      'cols':d_test['cols']
                      }
    #pdb.set_trace()
    return e_maps,d_new_test,d_new_train

def err_compare(d_old,d_new,err_maps,plot_dmap=False):
    '''
    Apply an error map to the centroid coordinates of a given list of images
    and return statistics of the error. Option to plot distortion map.
    
    inputs:
        img: list of image to test error map on ie ['mx2']
        err_map: error maps to apply to image
        dmap: set to True to plot resulting distortion map
        
    Outputs:
        err_info: various info/stats on error between modeled star pixel locations and
                   the centroid pixel locations that were 'corrected' by error map
                   
    Example Usage:
        1) create error map for 'mx' camera using images 'mx2' and 'mx3' and then
            apply error map to image 'mx1' and see how well the error is corrected
            
            import pandas as pd
            from err_maps import err_maps,err_compare
            err_map = err_maps(['mx1','mx2'],fit='2d polyfit',order=2)
            err_info = err_compare(['mx3'],err_map,plot_dmap=True)
    '''
    from dmap import dmap
    import pandas as pd
    
    img = d_old['img']
    err_info = {'x_err_old_'+img[0]:d_old['xdiff'],
                'y_err_old_'+img[0]:d_old['ydiff'],
                'mag_err_old_'+img[0]:d_old['mag'],
                'x_err_new_'+img[0]:d_new['xdiff'],
                'y_err_new_'+img[0]:d_new['ydiff'],
                'mag_err_new_'+img[0]:d_new['mag'],
                'x_img_'+img[0]:d_old['x_img'],
                'y_img_'+img[0]:d_old['y_img'],
                'x_act_'+img[0]:d_old['x_act'],
                'y_act_'+img[0]:d_old['y_act'],
                'x_img_new_'+img[0]:d_new['x_img'],
                'y_img_new_'+img[0]:d_new['y_img']
                }
    
    if plot_dmap == True:
        data_dmap = {'x_img_'+img[0]:d_old['x_img'],
                     'y_img_'+img[0]:d_old['y_img'],
                     'x_act_'+img[0]:d_old['x_act'],
                     'y_act_'+img[0]:d_old['y_act'],
                     'x_img_new_'+img[0]:d_new['x_img'],
                     'y_img_new_'+img[0]:d_new['y_img'],
                     'mag_err_old_'+img[0]:d_old['mag'],
                     'mag_err_new_'+img[0]:d_new['mag']}
        
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
    
    return err_info
    
    
    
    
    
    