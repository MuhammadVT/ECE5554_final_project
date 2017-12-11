# ECE5554 (Computer Vision) Final Project

[comment]: <> (<p align="center">
    <font size="+3">
    <b>
        Calibration of Satellite Cameras using Star Images <br>
    </font>
    <font size="+2">
        <br>
        Brett poche <br>
        <br> 
        Maimaitirebike (Muhammad Rafiq) Maimaiti <br>
    </b>
    </font>
</p>)

![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide1.PNG)

## Problem Statement
Polar Mesospheric Clouds (PMCs) form high up in the atmosphere near the polar mesopause (~82 km) and consist of tiny ice particles with sizes on the order of 100 nm [[1](https://www.sciencedirect.com/science/article/pii/S1364682608002861)]. The Cloud Imaging and Particle Size instrument (CIPS) onboard the Aeronomy of Ice (AIM) satellite has been recording images of PMCs in a polar orbit ever since launch in 2007. The CIPS instrument consist of four separate CCD imagers, and although the CCDs were calibrated prior to launch, science images obtained during orbit contain systematic flat-fielding errors. It is suspected that these errors could be due to unaccounted for distortion in the cameras.
Soon after launch in 2007, each of the four cameras in the CIPS instrument was pointed away from the Earth and recorded high-resolution images (14 images total, 2-3 per camera) of stars for calibration purposes. While current CIPS science retrieval algorithms utilize a simplistic pin-hole camera model, the goal of this project is to improve the camera model and reduce the existing systematic errors.   
![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide1.PNG)

## Approach
**Step 1)** Identify the star ‘names’ and RA/DEC coordinates in each star image using an automated star location software AstroImageJ. AstroImageJ takes an image of stars as an input and attempts to match/locate all stars in the image using a predefined star database and optimized search algorithm. Create a database of all stars located in the high resolution images with their associated RA/DEC coordinates.
![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide9.PNG)

**Step 2)** Write a module in Python to transform from world coordinates (RA/DEC) to image coordinates using the existing pinhole camera model. We use this module to take actual star locations (RA/DEC coordinates) from the star database created in step 1 as an input and output the expected precise sub-pixel location in image coordinates based on the pinhole camera model. These coordinates are referred to as “reference” coordinates.
![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide11.PNG)
**Step 3)** Use a centroid algorithm to calculate the sub-pixel location of all stars in the high resolution images which will be referred to as the “distorted” coordinates. There are 2-3 high resolution images per camera (4 cameras total) and each image contains ~60-100 stars.

**Step 4)** Determine the pixel offsets between the “reference” star coordinates and the “distorted” coordinates. This will result in a set of “pixel offsets” corresponding to a limited set of pixel coordinates on each image.
![](https://github.com/MuhammadVT/ECE5554_final_project/blob/master/CIPS_presentation_final_blank_background/Slide18.PNG)

**Step 5)** Ultimately we would like to use the resulting discrete “pixel offsets” found in Step 4 for each image and model the distortion in each camera. This final step will result in a camera model matrix that will transform every pixel location for each camera to a revised location that accounts for the distortion found in each camera.

## Error Modeling

## Evaluation

Due to a limited dataset (14 images total, 2-3 per camera) our approach of evaluation is (1) select one image from each camera for testing purposes (2) use the remaining images to create distortion models for each camera (3) quantify the resulting error on the test images. This method will allow us to determine the efficacy of our distortion models.

## Summary and Conclusions

## Installation
The codes in this project are develped in Windows10 and Ubuntu 16.04.3 LTS.
Here are the instructions for how to setup an conda environmet and execute the codes:

Use the Terminal or an Anaconda Prompt for the following steps.

#### Create the environment from the environment.yml file:

conda env create -f environment.yml

#### Activate the new environment:

Windows: activate cv_project 
macOS and Linux: source activate cv_project

#### Verify that the new environment was installed correctly:

conda list

#### Run the Jupyter Notebook

jupyter notebook






