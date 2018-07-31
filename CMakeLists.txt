cmake_minimum_required( VERSION 2.8 )
project( SFM )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

############### dependencies ######################
#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# pcl 
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#eigen
include_directories( "/usr/include/eigen3" )

set(SOURCE_FILES 
	src/main.cpp
	src/Image_Processing.h
	src/Image_Processing.cpp
	src/Math.h
	src/Math.cpp
	src/Bundle_Adjustment.h
	src/Bundle_Adjustment.cpp
	src/PointCloud_sfm.h
	src/PointCloud_sfm.cpp
)

add_executable(${PROJECT_NAME}  ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME}  ${OpenCV_LIBS} ${PCL_LIBRARIES})
