#include <iostream>
#include "Bundle_Adjustment.h"

using namespace std;

Bundle_Adjustment::Bundle_Adjustment(){ cout<<"Construct Bundle Adjustment"<<endl; }
Bundle_Adjustment::~Bundle_Adjustment(){}

void Bundle_Adjustment::Nonlinear_Triangulation(
			int num_iterate, vector<Point3f> total_pnts3D, 
 			vector<vector<int>> pnts3D_index, vector<vector<KeyPoint>> keypoints_all, 
 			Mat K, vector<Mat> Rotation, vector<Mat> Translation, 
 			vector<Point3f>& total_pnts3D_optimal)
{

    int num_pnt3D = total_pnts3D.size();
    for(int i=0; i<num_pnt3D; i++)
    {
        vector<Point2f> imgpnts;
        vector<int> frame_idx;
        Find3DpntCorrespondAllImgPnt(i, pnts3D_index, keypoints_all, imgpnts,frame_idx);

        vector<Mat> R_, T_;
        for(int j=0; j<frame_idx.size(); j++)
        {
            R_.push_back(Rotation[frame_idx[j]]);
            T_.push_back(Translation[frame_idx[j]]);
        }

        Point3f pnt3D_re = total_pnts3D[i];

        for(int k=0; k<num_iterate; k++){
            pnt3D_re = SinglePointNonlinear_Triangulation(
                                        pnt3D_re, imgpnts, K, R_, T_);
        }
        total_pnts3D_optimal.push_back(pnt3D_re);
    }
}

/*
Private
*/
Eigen::MatrixXd Bundle_Adjustment::JacobianMatrix(Point3f pnt3D, Mat K, Mat R, Mat T)
{

	Eigen::MatrixXd J(2,3);

    double u, v, w;      
    double f = 0.5*(K.at<double>(0) + K.at<double>(4));
    double px = K.at<double>(2);
    double py = K.at<double>(5);
    
    Eigen::MatrixXd K_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd T_;
    Eigen::MatrixXd Proj(3,4);
    Eigen::MatrixXd X(4, 1);
    Eigen::MatrixXd x(3, 1);

    Eigen::MatrixXd dev_u_X(1,3);
    Eigen::MatrixXd dev_v_X(1,3);
    Eigen::MatrixXd dev_w_X(1,3);

    Eigen::MatrixXd wdev_u_X(1,3);
    Eigen::MatrixXd wdev_v_X(1,3);
    Eigen::MatrixXd udev_w_X(1,3);
    Eigen::MatrixXd vdev_w_X(1,3);


    X(0,0) = pnt3D.x;
    X(1,0) = pnt3D.y;
    X(2,0) = pnt3D.z;
    X(3,0) = 1;

    K_ = MatrixTransform(K);
    R_ = MatrixTransform(R);
    T_ = MatrixTransform(T);

    Proj<<R_(0,0), R_(0,1), R_(0,2), T_(0,0),
          R_(1,0), R_(1,1), R_(1,2), T_(1,0),
          R_(2,0), R_(2,1), R_(2,2), T_(2,0);

    Proj = K_ * Proj;

    x = Proj * X;

    u = x(0,0);
    v = x(1,0);
    w = x(2,0);

    dev_u_X(0,0) = f*R_(0,0)+px*R_(2,0); 
    dev_u_X(0,1) = f*R_(0,1)+px*R_(2,1);
    dev_u_X(0,2) = f*R_(0,2)+px*R_(2,2);
    
    dev_v_X(0,0) = f*R_(1,0)+py*R_(2,0); 
    dev_v_X(0,1) = f*R_(1,1)+py*R_(2,1);
    dev_v_X(0,2) = f*R_(1,2)+py*R_(2,2);
    
    dev_w_X(0,0) = R_(2,0);
    dev_w_X(0,1) = R_(2,1);
    dev_w_X(0,2) = R_(2,2);

    wdev_u_X = w*dev_u_X;
    udev_w_X = u*dev_w_X;

    wdev_v_X = w*dev_v_X;
    vdev_w_X = v*dev_w_X;

    Eigen::MatrixXd J_row1(1,3);
    J_row1 = (wdev_u_X-udev_w_X)/(w*w);

    Eigen::MatrixXd J_row2(1,3);
    J_row2 = (wdev_v_X-vdev_w_X)/(w*w);

    J.row(0) = J_row1;
    J.row(1) = J_row2;

    return J;
}

Point3f Bundle_Adjustment::SinglePointNonlinear_Triangulation(
			Point3f pnt3D, vector<Point2f> imgPnts, Mat K, vector<Mat> R, vector<Mat> T)
{

    int num_imgPnts = imgPnts.size();

    Point3f pnt3D_next;
    Eigen::MatrixXd deltaX(3,1);
    Eigen::MatrixXd J(num_imgPnts*2,3);
    Eigen::MatrixXd b(num_imgPnts*2,1);
    Eigen::MatrixXd f(num_imgPnts*2,1);
    vector<Point3f> pnt3D_vec;
    pnt3D_vec.push_back(pnt3D);

    for(int i=0; i<num_imgPnts; i++)
    {
        Eigen::MatrixXd j(2,3);
        j = JacobianMatrix(pnt3D, K, R[i], T[i]);
        J.row(i*2) = j.row(0);
        J.row(i*2+1) = j.row(1);
    
        b(i*2,0) = imgPnts[i].x;
        b(i*2+1,0) = imgPnts[i].y;

        Mat rvec;
        vector<Point2f> projectedPoint;
        Rodrigues(R[i],rvec);
        projectPoints(pnt3D_vec, rvec, T[i], K, Mat(), projectedPoint);
    
        f(i*2,0) = projectedPoint[0].x;
        f(i*2+1,0) = projectedPoint[0].y;
    }

    Eigen::MatrixXd JtJ = J.transpose()*J;
    deltaX = JtJ.inverse()*J.transpose()*(b-f);

    pnt3D_next.x = pnt3D.x + deltaX(0,0);
    pnt3D_next.y = pnt3D.y + deltaX(1,0);
    pnt3D_next.z = pnt3D.z + deltaX(2,0);

    return pnt3D_next;
}

void Bundle_Adjustment::Find3DpntCorrespondAllImgPnt(
			int pnt_idx_3D, vector<vector<int>> pnts3D_index,
			vector<vector<KeyPoint>> keypoints_all, vector<Point2f>& imgpntAll,
			vector<int>& frame_idx)
{

    imgpntAll.clear();
    frame_idx.clear();
    
    for(int i=0; i<pnts3D_index.size(); i++)
    {
        for(int j=0; j<pnts3D_index[i].size(); j++)
        {
            if(pnts3D_index[i][j] == pnt_idx_3D)
            {
                imgpntAll.push_back(keypoints_all[i][j].pt);
                frame_idx.push_back(i);
                break;  
            }
        }
    }
}

Eigen::MatrixXd Bundle_Adjustment::MatrixTransform(Mat A_)
{
	int row = A_.rows;
	int col = A_.cols;

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