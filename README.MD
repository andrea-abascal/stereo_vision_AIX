Abstract:

This project presents the development of a functional stereo vision system prototype that allows the 3D reconstruction of scenes. Usually, off-the-shelf stereo vision systems don't allow the user to customize for their complex needs, they also tend to be expensive and there are certain restrictions regarding the camera resolution or a fixed baseline value that may promote noise in the stereo maps, therefore the 3D reconstructions end up being very limited and of unwanted quality. 

Computer vision algorithms are applied to real-time video capture in order to compute geometric estimates that allow to determine the space coordinates of an object in a scene, and thus creating a 3D reconstruction equivalent to the space observed. Furthermore, an intelligent algorithm is used to reduce noise and deliver a 3D reconstruction with few errors.  On the other hand, CAD software and 3D printing are used to develop the case that keeps the hardware that builds the stereo system. Also a baseline modifier is added in between cameras to explore its effects on the space coordinates estimations.

The result is a portable device with an embedded stereo vision system and the image  pre-, post-processing software for the scene reconstruction. Also the required documentation is generated to understand the functioning of the system and to analyze the effects of the results obtained in the different tests. This provides the user with a functional prototype that has better performance (improvements in image post-processing) and a lower cost of a customized system.


Once the set up is done the folowing progrmas must be runned:

[1] img_cap.py - This program is executed just once and it is meant to capture and store the calibration images.

    - This project includes syncronized images of a calibration board of 11 x 8 with 20 mm square sides. Pictures where taken with natural light.

[2] single_stereo_calibration.py - This program is executed just once and it is meant to carry out single camera calibration and stereo calibration. The resulting parameters are stored in the calibrationParameters.xml file. The following variables must be change according to the calibration setup.

    - chessboardSize = Calibration chessboard inner corners (row, column)

    - frameSize: Frame resolution

[3] stereo_rectify.py - This program is executed just once and it is meant to undistort and rectify images. The resulting rectification map is stored in the stereoMap.xml

[4] disparity_param.py - This program is executed just once and it is meant to tune the differnt parameters that may affect the disparity map. The resulting rectification map is stored in the disparity_map_params.xml. A brief description of the parameters is mentioned below:

    - Number of disparities (numDisparities): Sets the range of disparity values to be searched. Increasing the number of disparities increases the accuracy of the disparity map.

    - Block size (blockSize): Size of the sliding window used for block matching to find corresponding pixels in a rectified stereo image pair.

    - Pre-Filter Type (preFilterType): Enhance the texture information and improves the results of the block-matching algorithm. The filter type can be CV_STEREO_BM_XSOBEL or CV_STEREO_BM_NORMALIZED_RESPONSE.

    - Pre-filter size (preFilterSize): Window size of the filter used in the pre-filtering stage

    - Pre-filter cap (preFilterCap): Limits the filtered output to a specific value.
    
    - Minimum disparity (minDisparity): The minimum value of the disparity to be searched. In most scenarios it is set to zero.
    
    - Texture threshold (textureThreshold): Filters out areas that do not have enough texture information for reliable matching.
    
    - Uniqueness Ratio (uniquenessRatio): Another post-filtering step. The pixel is filtered out if the best matching disparity is not sufficiently better than every other disparity in the search range. 
    
    - Speckle range (speckleRange) and speckle window size (speckleWindowSize): The speckle range defines how close the disparity values should be to be considered as part of the same blob. The speckle window size is the number of pixels below which a disparity blob is dismissed as “speckle”.
    
    - disp12MaxDiff: Pixels are matched both ways, from the left image to the right image and from the right image to left image. disp12MaxDiff defines the maximum allowable difference between the original left pixel and the back-matched pixel.

[5] depth_estimation.py - 
[6] 3d_reconstruction.py - 


Test Programs

The following programs can be executed to run tests or verify some results. In order to run the tests the corresponding files must be obtained previously.

[*] rectification_test.py - This programs verifies that both camera streams are rectified (epipolar lines match). It displays both streams on a single window and green lines are drawn. The calibration board can be used to verify that lines pass through the same points. This program uses stereoMap.xml file.

[*] disparity_test.py - This programs displays the disparity map using the previously tuned parameters. This program uses disparity_map_params.xml file.