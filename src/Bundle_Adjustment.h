#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

class Bundle_Adjustment
{
public:
    Bundle_Adjustment();
    ~Bundle_Adjustment();

 	void Nonlinear_Triangulation(int num_iterate,
 								 vector<Point3f> total_pnts3D, 
 								 vector<vector<int>> pnts3D_index,
 								 vector<vector<KeyPoint>> keypoints_all, Mat K,
 								 vector<Mat> Rotation, vector<Mat> Translation,
 								 vector<Point3f>& total_pnts3D_optimal);

private:
	Eigen::MatrixXd JacobianMatrix(Point3f pnt3D, Mat K, Mat R, Mat T);

	Point3f SinglePointNonlinear_Triangulation(Point3f pnt3D, vector<Point2f> imgPnts, 
											   Mat K, vector<Mat> R, vector<Mat> T);

	void Find3DpntCorrespondAllImgPnt(int pnt_idx_3D,
 	 							      vector<vector<int>> pnts3D_index,
 	 							      vector<vector<KeyPoint>> keypoints_all,
 								      vector<Point2f>& imgpntAll,
 								      vector<int>& frame_idx);

	Eigen::MatrixXd MatrixTransform(Mat A_);

};
#endif