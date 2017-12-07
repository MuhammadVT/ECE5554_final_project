# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:29:05 2017

@author: addiewan
"""
import os

def gen_stars(cen_coord_loc=os.getcwd() + '/data/img_pixel_coords/',
              save_loc=None,
              model_compare=False):
    '''
    Generate star database
    
    Input:
        cen_coord_loc: folder location of centroid pixel locations for each image
        save_stars_loc: (optional) file path to save the database to 
        model_compare: create x_act_model_img and y_act_model_img columns by running each
                       ra/dec value through the python radec2pixel.py module
        
    Output:
        data: pandas dataframe containing star database
        
    Example Usage:
        
        1) Generate a star database
        loc = os.getcwd() +'/img_pixel_coords/'
        data = gen_stars(loc)
        
        2) Generate and save a star database
    '''
    import os
    import numpy as np
    import pandas as pd
    from glob import glob
    from radec2pixel import radec2pixel,radec2v_global
    from star_utils import unit_vector,angle_between
    
    data_loc = glob(cen_coord_loc + '*.txt')   
    data={}
    for file in data_loc:
        file = file.replace('\\','/')
        #print(file)
        data[file[-7:-4]] = pd.read_csv(file)

    img_list = list(data.keys())
    data = pd.concat([data[img_list[0]],
                      data[img_list[1]],
                      data[img_list[2]],
                      data[img_list[3]],
                      data[img_list[4]],
                      data[img_list[5]],
                      data[img_list[6]],
                      data[img_list[7]],
                      data[img_list[8]],
                      data[img_list[9]],
                      data[img_list[10]],
                      data[img_list[11]]],
                      axis=0, ignore_index=False)   
    data_all = data.copy()    
    #Currentely the RA/DEC decimals are limited to 3 due to inconsistencies among different datafiles
    #Should see if this amount of precision is sufficient
    data = data.round(3)              
    data = data.groupby(['ra_act', 'dec_act']).first().reset_index()
    
    data_all = data_all.set_index('starname')
    data_all = data_all.sort_index()
    
    data = data.set_index('starname')
    data = data.sort_index()
    
    data['v_global_x'] = np.nan  
    data['v_global_y'] = np.nan
    data['v_global_z'] = np.nan  
    for i in range(len(data.index)):
        v_global = radec2v_global(data.loc[data.index[i],['ra_act']].values,
                                  data.loc[data.index[i],['dec_act']].values)
        v_global = unit_vector(v_global)
        
        data.loc[data.index[i],'v_global_x'] = v_global[0][0]
        data.loc[data.index[i],'v_global_y'] = v_global[1][0]
        data.loc[data.index[i],'v_global_z'] = v_global[2][0]
    
    if model_compare == True:
        for img in img_list:
            data['x_act_model_'+img]=np.nan
            data['y_act_model_'+img]=np.nan
        for i in range(len(data.index)):
            #print(i,data.index[i])
            for img in img_list:
                output = radec2pixel(data.loc[data.index[i],['ra_act']].values,
                                 data.loc[data.index[i],['dec_act']].values,
                                 img,
                                 img[0:2],
                                 proj='1a')
                if np.isnan(data.loc[data.index[i],'x_act_'+img]) == False:
                    data.loc[data.index[i],'x_act_model_'+img] = output['x'][0]
                    data.loc[data.index[i],'y_act_model_'+img] = output['y'][0]
    
    x_img = 'x_img_'
    y_img = 'y_img_'
    x_act = 'x_act_'
    y_act = 'y_act_'
    offset = 759.5
    
    for img in img_list:
        data['err_mag_'+img] = np.sqrt(((data[x_img+img]-offset)-(data[x_act+img]-offset))**2 + ((data[y_img+img]-offset)-(data[y_act+img]-offset))**2)
        data['err_r_'+img] = np.sqrt((data[x_img+img]-offset)**2 + (data[y_img+img]-offset)**2)
        for i in range(len(data.index)):
            data.loc[data.index[i],'err_ang_'+img] = angle_between(vectors_img[i],vectors_err[i])*180.0/np.pi
    
    
    if save_loc != None:
        data.to_csv(save_stars_loc)
        
    return data