#ifndef MATH_H
#define MATH_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

class Math
{
public:
    Math();
    ~Math();

	vector<Point3f> Triangulation(
		Mat K,Mat R1, Mat T1, Mat R2,Mat T2, 
		vector<Point2f> pnts1, vector<Point2f> pnts2);	

	vector<Point2f> reprojection(
		vector<Point3f> pnts3D_this_frame,  
    	Mat K, Mat Rotation, Mat Translation);    


 	vector<Point3f> triangulation_opencv (
	 	const vector<Point2f>& pnts1, 
		const vector<Point2f>& pnts2,
    	const Mat& K, const Mat& R, const Mat& t);


private:
	Eigen::MatrixXd MatrixTransform(Mat A_, int row, int col);
	Eigen::MatrixXd PointsToMatrix(vector<Point2f> pnts);
	Eigen::MatrixXd PointsToMatrix(vector<Point3f> pnts);
	vector<Point2f> MatrixToPoints(Eigen::MatrixXd A);

	Point2f pixel2cam ( const Point2f& p, const Mat& K );

};
#endif //MATRIX_H