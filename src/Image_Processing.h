#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include "Math.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

struct ImgProConfig
{
	cv::Mat K;
	cv::Mat distCoeffs;
	int SURFminHessian;
	float MatchminRatio;
};

class Image_Processing
{
public:
	Image_Processing(ImgProConfig config_);
	~Image_Processing();

	void GetFeatureSURFAll(vector<Mat> imgs, 
						   vector<vector<KeyPoint>>& keypoints_all, 
		                   vector<Mat>& descriptors_all, 
		                   vector<vector<Vec3b>>& colors_all);	

	void MatchFeatureAll(vector<Mat> descriptors_all,
						 vector<vector<DMatch>>& matches_all);

	//Find Rotation Translation of Frame2 , points3D and its index
 	void Init_Construction(vector<KeyPoint> keypoints_first_frame, 
 						   vector<KeyPoint> keypoints_second_frame,
 						   vector<DMatch> matches,
 						   vector<Vec3b> colors_first_frame, 
 						   vector<Vec3b>& total_colors,
		                   vector<Point3f>& total_pnts3D,
		                   vector<vector<int>>& pnts3D_index,
                           Mat& Rot_second_frame, 
                           Mat& Trans_second_frame);

 	//Do PNP find New Rotation and Translation, new pnts3D
 	void New_Construction(vector<KeyPoint> keypoints_last_frame, 
 						  vector<KeyPoint> keypoints_new_frame,
 						  vector<DMatch> matches,
 						  vector<Vec3b> colors_last_frame,
 						  Mat Rot_last_frame,
 						  Mat Trans_last_frame,
 						  vector<Point3f> total_pnts3D,
 						  vector<int> pnts3D_index_last_frame,
 						  vector<Vec3b>& new_colors,
 						  vector<Point3f>& new_pnts3D,
 						  Mat& Rot_new_frame, 
		                  Mat& Trans_new_frame);

 	//update pnts3D, pnts3D_index, colors
 	void Fusion_Construction(vector<DMatch> matches,
                             vector<Point3f> new_pnts3D,
                             vector<Vec3b> new_colors,
                             vector<int>& pnts3D_index_last_frame,
                             vector<int>& pnts3D_index_new_frame,
                             vector<Point3f>& total_pnts3D,
                             vector<Vec3b>& total_colors);

 	void FindCorrespond3DandKP(vector<KeyPoint> keypoints_this_frame, 
 							   vector<Point3f> total_pnts3D, 
 					  		   vector<int> pnts3D_index,
 					           vector<Point3f>& pnts3D_this_frame,
    						   vector<Point2f>& pntsKP_to_3D);

 	void ReprojectionError(vector<Point3f> pnts_3D, vector<Point2f> pnts_img,
 						   Mat Rotation, Mat Translation,
 						   vector<Point2f>& pnts_repro, double& error);

 	Point3f rotationMatrixToEulerAngles(Mat &R);


private:
	void GetFeatureSURF(Mat img, vector<KeyPoint>& keypoints, 
						Mat& descriptors, vector<Vec3b>& colors);

	void MatchFeatures(Mat query, Mat train, vector<DMatch>& matches);

	bool Find_Transform_RT(vector<Point2f> p1, vector<Point2f> p2, Mat K,
					       Mat& Rot, Mat& Trans, Mat& mask);

	void GetMatchPointLocation(vector<KeyPoint> kp1,vector<KeyPoint> kp2, 
							   vector<DMatch> matches,
                               vector<Point2f>& pnt1, vector<Point2f>& pnt2);
	//get color only queryIdx
	void GetMatchPointColor(vector<Vec3b> colors,
                    		vector<DMatch> matches,
                    		vector<Vec3b>& Outcolors_queryIdx);
	
	//get color both trainIdx and queryIdx
	void GetMatchPointColor(vector<Vec3b> colors1, vector<Vec3b> colors2, 
                    		vector<DMatch> matches,
                    		vector<Vec3b>& Outcolors1, vector<Vec3b>& Outcolors2);
	
	bool isRotationMatrix(Mat &R);

	vector<Point2f> reprojection_with_distort(
							     vector<Point3f> pnts3D_this_frame,  
    					         Mat K, Mat distCoeffs, Mat Rotation, Mat Translation);



private:
	Math* math;
	cv::Mat K;
	cv::Mat distCoeffs;
	int SURFminHessian;
	float MatchminRatio;
};
#endif