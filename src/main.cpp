#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Image_Processing.h"
#include "Bundle_Adjustment.h"
#include "PointCloud_sfm.h"

using namespace std;
using namespace cv;

void DrawCircle(Mat Img, vector<Point2f> pnts1, vector<Point2f> pnts2, int index);

int main()
{
    VideoCapture cap("images/%d.jpg");
    vector<Mat> imgs;
    
    while( cap.isOpened() )
    {
        Mat img;
        cap.read(img);
        if (img.empty()){
          cout << "End of Sequence" << endl;
          cout << "Total Image: "<< imgs.size() << endl;
          break;
        }
        imgs.push_back(img);  
    }


    Mat K(3, 3, CV_64F);
    K.at<double>(0.0) = 618.854; K.at<double>(0,1) = 0;       K.at<double>(0,2) = 320;
    K.at<double>(1,0) = 0;       K.at<double>(1,1) = 618.854; K.at<double>(1,2) = 240;
    K.at<double>(2,0) = 0;       K.at<double>(2,1) = 0;       K.at<double>(2,2) = 1;

    // k1, k2, p1, p2, k3
    Mat distCoeffs(5,1,cv::DataType<double>::type);
    distCoeffs.at<double>(0) = -1.0325128317185285e-01;   
    distCoeffs.at<double>(1) = 1.6735783025915860e-01;
    distCoeffs.at<double>(2) = 0;
    distCoeffs.at<double>(3) = 0;
    distCoeffs.at<double>(4) = -4.0206256447312419e-02;

    ImgProConfig config;
    config.K = K;
    config.distCoeffs = distCoeffs;
    config.SURFminHessian = 400;
    config.MatchminRatio = 0.7;

    Image_Processing imgpro(config);

    vector<Mat> Rotation;
    vector<Mat> Translation;
    Mat R0 = Mat::eye(3, 3, CV_64FC1);
    Mat T0 = Mat::zeros(3, 1, CV_64FC1);
    Rotation.push_back(R0);
    Translation.push_back(T0);



    //get feature all
    vector<vector<KeyPoint>> keypoints_all; 
    vector<Mat> descriptors_all; 
    vector<vector<Vec3b>> colors_all;
    imgpro.GetFeatureSURFAll(imgs, keypoints_all, descriptors_all, colors_all);



    //Match feature all
    vector<vector<DMatch>> matches_all;	  
    imgpro.MatchFeatureAll(descriptors_all, matches_all);



    //init construction
    vector<vector<int>> pnts3D_index;               //same match index is the same
    pnts3D_index.resize(keypoints_all.size());
    for(int i=0; i<pnts3D_index.size(); i++)
        pnts3D_index[i].resize(keypoints_all[i].size(), -1);  //all value init = -1

    vector<Vec3b> total_colors;
    vector<Point3f> total_pnts3D;

    Mat R1, T1;
    imgpro.Init_Construction(keypoints_all[0],keypoints_all[1],
                             matches_all[0], colors_all[0],
                             total_colors, total_pnts3D, pnts3D_index, R1, T1);
    Rotation.push_back(R1);
    Translation.push_back(T1);



    //from second match to last match
    for(int i=1; i<matches_all.size(); i++)   
    {

      vector<Vec3b> new_colors;
      vector<Point3f> new_pnts3D;

      Mat R, T;
      imgpro.New_Construction(keypoints_all[i], keypoints_all[i+1],
                              matches_all[i], colors_all[i],
                              Rotation[i],Translation[i],
                              total_pnts3D, pnts3D_index[i], 
                              new_colors, new_pnts3D, R, T);

      Rotation.push_back(R);
      Translation.push_back(T);

      imgpro.Fusion_Construction(matches_all[i], new_pnts3D, new_colors,
                                 pnts3D_index[i],pnts3D_index[i+1], 
                                 total_pnts3D, total_colors);
    }

    for(int i=0; i<Rotation.size(); i++)
    {
        cout<<"Rotation "<<i<<" :"<<endl;
        cout<<Rotation[i]<<endl;
        cout<<"Translation "<<i<<" :"<<endl;
        cout<<Translation[i]<<endl;

    }
    
    cout<<"Number of 3D Points: "<<total_pnts3D.size()<<endl;

    int num_imgs = imgs.size();

    //Reprojection
    for(int i=0; i<imgs.size(); i++)
    {

        vector<Point3f> pnts3D_this_frame;
        vector<Point2f> pntsimg_to_3D;
        imgpro.FindCorrespond3DandKP(keypoints_all[i],total_pnts3D, pnts3D_index[i],
                                     pnts3D_this_frame, pntsimg_to_3D);


        vector<Point2f> pnt_repro;
        double repro_error;
        imgpro.ReprojectionError(pnts3D_this_frame, pntsimg_to_3D, 
                                 Rotation[i], Translation[i],
                                 pnt_repro, repro_error);
        double average_error = repro_error/pnts3D_this_frame.size();

        cout<<"Average reprojection Error: "<<average_error<<endl;
        DrawCircle(imgs[i],pntsimg_to_3D, pnt_repro, i);
        waitKey(0); 
    }



    //bundle adjustmemt
    Bundle_Adjustment bj;
    int num_iterate = 3;
    vector<Point3f> total_pnts3D_optimal;
    bj.Nonlinear_Triangulation(
                      num_iterate, total_pnts3D, pnts3D_index, keypoints_all, K, 
                      Rotation, Translation, total_pnts3D_optimal);

    for(int i=0; i<total_pnts3D_optimal.size(); i++)
    {
        double res = cv::norm(total_pnts3D[i]-total_pnts3D_optimal[i]);
        if(res>1000)
          total_pnts3D_optimal[i] = total_pnts3D[i];
    }

    //reprojection after bundle adjustment
    for(int i=0; i<imgs.size(); i++)
    {

        vector<Point3f> pnts3D_this_frame;
        vector<Point2f> pntsimg_to_3D;
        imgpro.FindCorrespond3DandKP(keypoints_all[i],total_pnts3D_optimal, pnts3D_index[i],
                                     pnts3D_this_frame, pntsimg_to_3D);


        vector<Point2f> pnt_repro;
        double repro_error;
        imgpro.ReprojectionError(pnts3D_this_frame, pntsimg_to_3D, 
                                 Rotation[i], Translation[i],
                                 pnt_repro, repro_error);

        double average_error = repro_error/pnts3D_this_frame.size();

        cout<<"Average reprojection Error: "<<average_error<<endl;
        DrawCircle(imgs[i],pntsimg_to_3D, pnt_repro, i);
        waitKey(0); 
    }


    //point cloud
    PointCloud_sfm pcl_sfm;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr;
    point_cloud_ptr = pcl_sfm.GeneratePointCloud(total_pnts3D, total_colors);
    //point_cloud_ptr = pcl_sfm.GeneratePointCloud(total_pnts3D_optimal, total_colors);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = pcl_sfm.rgbVis(point_cloud_ptr, Rotation, Translation);

  	//--------------------
  	// -----Main loop-----
  	//--------------------
  	while (!viewer->wasStopped ())
  	{
    	viewer->spinOnce (100);
    	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  	}

	waitKey(0); 

    return 0;
}

void DrawCircle(Mat Img, vector<Point2f> pnts1, vector<Point2f> pnts2, int index)
{
    for(int i=0; i<pnts1.size(); i++)
    {
        circle(Img, pnts1[i], 2, Scalar(0,0,255), -1);
    }

    for(int i=0; i<pnts2.size(); i++)
    {
        circle(Img, pnts2[i], 2, Scalar(255,0,0), -1);
    }

    string filename;
    filename = std::to_string(index)+ ".jpg";
    imwrite (filename, Img);

    imshow("window", Img);
}