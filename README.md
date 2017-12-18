![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide1.PNG)

## Problem Statement
Polar Mesospheric Clouds (PMCs) form high up in the atmosphere near the polar mesopause (~82 km) and consist of tiny ice particles with sizes on the order of 100 nm [[1](https://www.sciencedirect.com/science/article/pii/S1364682608002861)]. The Cloud Imaging and Particle Size instrument (CIPS) onboard the Aeronomy of Ice (AIM) satellite has been recording images of PMCs in a polar orbit ever since launch in 2007. The CIPS instrument consist of four separate CCD imagers, and although the CCDs were calibrated prior to launch, science images obtained during orbit contain small but significant systematic flat-fielding errors. It is suspected that these errors could be due to unaccounted for distortion in the cameras.
Soon after launch in 2007, each of the four cameras in the CIPS instrument was pointed away from the Earth and recorded high-resolution images (14 images total, 2-3 per camera) of stars for calibration purposes. The goal of this project is to use these star images to quantify distortion in CIPS cameras and reduce the existing systematic errors.   
![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide4.PNG)



## Approach
**Overview:** The general approach is to input a raw star image and use an ideal camera model to produce a distortion map. This distortion map is then used to generate a 'error offset' model that can be applied to existing science data.

![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Goal_input_to_output.PNG)

**Step 1)** Identify the star ‘names’ and star coordinates (J2000 RA/DEC) [[2]](http://astronomy.swin.edu.au/cosmos/E/Epoch) in each star image using an automated star location software AstroImageJ [[3](http://www.astro.louisville.edu/software/astroimagej/)]. AstroImageJ takes an image of stars as an input and attempts to match/locate all stars in the image using a predefined star database and optimized search algorithm. A database is then generated using all of the stars located in the high resolution images with their associated RA/DEC coordinates.

![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Step2_ra_dec_to_pixel_coords_combined.PNG)

**Step 2)** Write a module in Python to transform from star coordinates (RA/DEC) to image coordinates. The transformation from star coordinates to camera coordinates is achieved using spherical geometry and quaternions provided by an onboard star tracker. The conversion from camera coordinates to image pixel coordinates uses an ideal pinhole camera model. We use this module to take actual star locations (RA/DEC coordinates) from the star database created in step 1 as an input and output the expected precise sub-pixel location in image coordinates based on the pinhole camera model. These coordinates are referred to as “reference” coordinates.

![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Step2a_coord_transform_R2.PNG)

**Step 3)** Use a centroid algorithm to calculate the sub-pixel location of all stars in the high resolution images which will be referred to as the “distorted” coordinates. There are 2-3 high resolution images per camera (4 cameras total) and each image contains ~60-100 stars.

**Step 4)** Determine the pixel offsets between the “reference” star coordinates and the “distorted” coordinates. This will result in a set of “pixel error offsets” corresponding to a limited set of pixel coordinates on each image.

![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Steps3_4_centroid_to_error_offset_R2.PNG)

**Step 5)** Ultimately we would like to use the resulting discrete “pixel error offsets” found in Step 4 for each image and model the distortion in each camera. This final step will result in a camera error mapping that will transform every pixel location for each camera to a revised location that accounts for the distortion found in each camera.

## Error Modeling

### Surface Fitting
In this approach we built three surface fitting models (Linear Interpolation, Nearest Neighbor, and Polynomial Fitting) to predict the magnitude of pixel error offsets in each star image. Please see the details in [this notebook](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/build_surface_fit_models.ipynb)

### Regression Models
In this approach we built three regression models (Linear Regression, Gradient Boosting Regression, and Random Forest Regression) to predict the error vectors (both magnitude and angle). The model results are then compared with a baseline model. Absolute error and squared error metrics are used to eveluate model predictions for both magnitudes and angles of the error vectors. Please see the details in [this notebook](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/build_regression_models.ipynb)


## Model Evaluation

Due to a limited dataset (14 images total, 2-3 per camera) our approach of evaluation is (1) select one image from each camera for testing purposes (2) use the remaining images to create distortion models for each camera (3) quantify the resulting error on the test images. This method will allow us to determine the efficacy of our distortion models.

### Evaluation of Surface Fitting
Here we show the results of applying the surface fitting error maps to selected datasets. [this notebook](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/surface_fit_evaluation.ipynb)

### Evaluation of Regression Model
Here we choose one of the three regression models to predict the error vectors and evaluation its performance by visualizing the expected and predicted error distortion maps. Please see the details in [this notebook](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/regression_model_evaluation.ipynb)




## Summary and Conclusions

The star identification using Astrometry.net was successful in a large number of star images, however there are multiple star images that Astrometry.net was unable to identify and required manual identification. The stars requiring manual identification resulted in far less stars per image. Even with the limited dataset, we were able to identify distortion in some cameras ranging from 0.5 to greater than 1.0 pixels across the fields of view. The shape and extent of the distortion maps are seen to vary from camera to camera, and the error models built using stars from the manually identified star images seem to perform the worst.

The figure below illustrates the overall effectiveness of the surface fitting error maps of decreasing the pixel error offset of star centroids for the 'PX' camera. The blue boxplot illustrates the spread of all error magnitudes of the original image or set of images included in the respective dataset. The green boxplots correspond to the result after linear interpolation, and the red boxplots correspond to the resulting error magnitudes after application of the nearest neighbor algorithm. 

![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/summary_stats_surface_fit.PNG)

Further analysis of the star centroid algorithm remains, and future work will entail quantifying the magnitude of uncertainty in star centroid calculations. While it appears that the distortion maps are effective at reducing the overall pixel offset across the field of view, the ultimate test will be to apply these distortion maps to CIPS science images and analyze the resulting effects. 


## References 
1.  W.E. McClintock, D.W. Rusch, G.E. Thomas, A.W. Merkel, M.R. Lankton, V.A. Drake, S.M. Bailey, J.M. Russell, The cloud imaging and particle size experiment on the Aeronomy of Ice in the mesosphere mission: Instrument concept, design, calibration, and on-orbit performance, In Journal of Atmospheric and Solar-Terrestrial Physics, Volume 71, Issues 3â€“4, 2009, Pages 340-355, ISSN 1364-6826, https://doi.org/10.1016/j.jastp.2008.10.011.
2.  http://astronomy.swin.edu.au/cosmos/E/Epoch
3.  http://www.astro.louisville.edu/software/astroimagej/


## Installation
The codes in this project are develped in Windows10 and Ubuntu 16.04.3 LTS.
Here are the instructions for how to setup an conda environmet and execute the codes:

Use the Terminal or an Anaconda Prompt for the following steps.

#### Create the environment from the environment.yml file:

*conda env create -f environment.yml*

#### Activate the new environment:

Windows: *activate cv_project*

macOS and Linux: *source activate cv_project*

#### Verify that the new environment was installed correctly:

*conda list*

#### Run the Jupyter Notebook

*jupyter notebook*






