Description:

This project presents the development of a functional stereo vision application prototype that allows the 3D reconstruction of scenes. 

Computer vision algorithms are applied to real-time video capture in order to compute geometric estimates that allow to determine the space coordinates of an object in a scene, and thus creating a 3D reconstruction equivalent to the space observed. Furthermore, post-processing techniques are used to reduce noise and deliver a 3D reconstruction with few errors.  

The result is a software package with programs used in image preprocessing and post-processing that can be implemented in any stereo configuration with the objective of developing a system that virtualizes a physical space. This provides the user with a functional software prototype that allows for 3D reconstruction of scenes with results equivalents to the comercial options. 

Once the set up is done the folowing programs must be runned:

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

[5] disparity_paramSGBM.py - This program uses the StereoMatch algorithm based in Semi Global. It is executed just once and it is meant to tune the different parameters that may affect the disparity map. The resulting rectification map is stored in the disparity_map_paramsSGBM.xml. A brief description of the parameters is shown above.


Test Programs

The following programs can be executed to run tests or verify some results. In order to run the tests the corresponding files must be obtained previously.

[*] rectification_test.py - This programs verifies that both camera streams are rectified (epipolar lines match). It displays both streams on a single window and green lines are drawn. The calibration board can be used to verify that lines pass through the same points. This program uses stereoMap.xml file.

[*] disparity_test.py - This program displays the disparity map using the previously tuned parameters. The user can choose the type of stereo matching algorithm. This program uses disparity_map_params.xml file or disparity_map_paramsSGBM.xml file.

[*] disparityWLSFilter_test.py - This program uses the desired stereo match algorithm chosen by the user, and it adds a WLS filter to video post processing the disparity map and enhances features.

[*] depth_Map.py - This program displays the depth map from the obtained disparity map with the WLS filter applied. It allows the user to evaluate performance with FPS calculation.

[*] 3D_reconstruction.py - This program uses the functionalities of previous programs to reproject 2D points into 3D points. These points and its associated RGB colors are stored in a PLY file pointcloud.ply.

[*] visualize_PCD.py - After generating a PLY file with the 3D points in the previous program, this program allows the visualization of the point cloud using Open3D.
