#include <iostream>
#include "Math.h"

using namespace std;

Math::Math(){ cout<<"Construct Math!!"<<endl; }

Math::~Math(){}


vector<Point3f> Math::Triangulation(
			Mat K,Mat R1, Mat T1, Mat R2,Mat T2, 
			vector<Point2f> pnts1, vector<Point2f> pnts2)
{
    Eigen::MatrixXd K_ = MatrixTransform(K,3,3);
    Eigen::MatrixXd R1_ = MatrixTransform(R1,3,3);
    Eigen::MatrixXd T1_ = MatrixTransform(T1,3,1);
    Eigen::MatrixXd R2_ = MatrixTransform(R2,3,3);
    Eigen::MatrixXd T2_ = MatrixTransform(T2,3,1);

    Eigen::MatrixXd Proj1(3,4);
    Eigen::MatrixXd Proj2(3,4);

    Proj1<<R1_(0,0), R1_(0,1), R1_(0,2), T1_(0,0),
           R1_(1,0), R1_(1,1), R1_(1,2), T1_(1,0), 
           R1_(2,0), R1_(2,1), R1_(2,2), T1_(2,0);

    Proj2<<R2_(0,0), R2_(0,1), R2_(0,2), T2_(0,0), 
           R2_(1,0), R2_(1,1), R2_(1,2), T2_(1,0),
           R2_(2,0), R2_(2,1), R2_(2,2), T2_(2,0);

    Proj1 = K_ * Proj1;
    Proj2 = K_ * Proj2;


    int num_points = pnts1.size();
    vector<Point3f> pnts3D;   //N
    pnts3D.reserve(num_points);

    for(int i=0; i<num_points; i++)
    {
        Eigen::MatrixXd x1(3,1);
        x1<<pnts1[i].x,
            pnts1[i].y,
            1;

        Eigen::MatrixXd x2(3,1);
        x2<<pnts2[i].x,
            pnts2[i].y,
            1;


        Eigen::MatrixXd x1_cross(3,3);
        Eigen::MatrixXd x2_cross(3,3);

        x1_cross<<0, -1,  x1(1,0),
                  1,  0, -x1(0,0),
                  -x1(1,0), x1(0,0), 0;

        x2_cross<<0, -1, x2(1,0),
                  1, 0, -x2(0,0),
                  -x2(1,0), x2(0,0), 0;


        Eigen::MatrixXd A1(3,4);
        Eigen::MatrixXd A2(3,4);

        A1 = x1_cross*Proj1;
        A2 = x2_cross*Proj2;

        //A = [A1;A2]
        Eigen::MatrixXd A(6,4);
        A<<A1(0,0), A1(0,1), A1(0,2), A1(0,3),
           A1(1,0), A1(1,1), A1(1,2), A1(1,3),
           A1(2,0), A1(2,1), A1(2,2), A1(2,3),
           A2(0,0), A2(0,1), A2(0,2), A2(0,3),
           A2(1,0), A2(1,1), A2(1,2), A2(1,3),
           A2(2,0), A2(2,1), A2(2,2), A2(2,3);

        //SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
    
        //normalize
        pnts3D.push_back(Point3f(V(0,3)/V(3,3), V(1,3)/V(3,3),  V(2,3)/V(3,3)));
    }

    return pnts3D;
}

vector<Point2f> Math::reprojection(vector<Point3f> pnts3D_this_frame,  
    					         Mat K, Mat Rotation, Mat Translation)
{

    vector<Point2f> pnts_repro;

    Eigen::MatrixXd pnts3D_ = PointsToMatrix(pnts3D_this_frame);
    Eigen::MatrixXd K_ = MatrixTransform(K,3,3);
    Eigen::MatrixXd R_ = MatrixTransform(Rotation,3,3);
    Eigen::MatrixXd T_ = MatrixTransform(Translation,3,1);
    Eigen::MatrixXd Proj(3,4);

    Proj<<R_(0,0), R_(0,1), R_(0,2), T_(0,0),
          R_(1,0), R_(1,1), R_(1,2), T_(1,0),
          R_(2,0), R_(2,1), R_(2,2), T_(2,0);

    Proj = K_ * Proj;

    Eigen::MatrixXd pnts_repro_ = Proj * pnts3D_;
    pnts_repro = MatrixToPoints(pnts_repro_);

    return pnts_repro;
}



vector<Point3f> Math::triangulation_opencv (
	 	const vector<Point2f>& pnts1, 
		const vector<Point2f>& pnts2,
    	const Mat& K, const Mat& R, const Mat& t)
{

    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    

    int size = pnts1.size();

    vector<Point2f> pts_1_cam, pts_2_cam;
    for ( int i=0; i<size; i++ )
    {
        // 将像素坐标转换至相机坐标
        pts_1_cam.push_back ( pixel2cam( pnts1[i], K) );
        pts_2_cam.push_back ( pixel2cam( pnts2[i], K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1_cam, pts_2_cam, pts_4d );
    

    vector<Point3f> points;   //N
    points.reserve(size);

    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3f p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }

    return points;
}


/*
//private
*/
Eigen::MatrixXd Math::MatrixTransform(Mat A_, int row, int col)
{

    Eigen::MatrixXd A(row, col);
    for(int i=0; i<row; i++)
    {
        for(int j=0;j<col;j++)
        {
            A(i,j) = A_.at<double>(i,j);
        }
    }
    return A;
}

Eigen::MatrixXd Math::PointsToMatrix(vector<Point2f> pnts)
{
    Eigen::MatrixXd x(3, pnts.size());

    for(int i=0; i<pnts.size(); i++)
    {
        x(0,i) = pnts[i].x;
        x(1,i) = pnts[i].y;
        x(2,i) = 1;
    }
    return x;
}

Eigen::MatrixXd Math::PointsToMatrix(vector<Point3f> pnts)
{
    Eigen::MatrixXd X(4, pnts.size());
    for(int i=0; i<pnts.size(); i++)
    {
        X(0,i) = pnts[i].x;
        X(1,i) = pnts[i].y;
        X(2,i) = pnts[i].z;
        X(3,i) = 1;
    }
    return X;
}

vector<Point2f> Math::MatrixToPoints(Eigen::MatrixXd A)
{
    vector<Point2f> pnts;
    for(int i=0; i<A.cols(); i++)
    {
        //normalize
        pnts.push_back(Point2f(A(0,i)/A(2,i),A(1,i)/A(2,i)));
    }
    return pnts;
}

// (u-Cx)/fx = X/Z
// (v-Cy)/fy = Y/Z
Point2f Math::pixel2cam ( const Point2f& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}