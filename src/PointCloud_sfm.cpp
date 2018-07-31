#include <iostream>
#include "PointCloud_sfm.h"

using namespace std;

PointCloud_sfm::PointCloud_sfm(){ cout<<"Construct PointCloud_sfm"<<endl; }

boost::shared_ptr<pcl::visualization::PCLVisualizer> PointCloud_sfm::rgbVis (
	pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
	vector<Mat> Rotations, vector<Mat> Translations)
{
  	// --------------------------------------------
  	// -----Open 3D viewer and add point cloud-----
  	// --------------------------------------------
  	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor (0, 0, 0);
  	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
 	  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  	viewer->addCoordinateSystem (1.0);
  	viewer->initCameraParameters ();

   	//------------------------------------
  	//-----Add coordinate system----
  	//------------------------------------
  	int number_frames = Rotations.size();

  	for(int i=0; i<number_frames; i++)
  	{
  		Mat C = Translations[i];
    	pcl::PointXYZ origin;
  		pcl::PointXYZ x_axis, y_axis, z_axis;

  	    origin.x = C.at<double>(0);
  		origin.y = C.at<double>(1);
  		origin.z = C.at<double>(2);

  		x_axis.x = origin.x + Rotations[i].at<double>(0);
  		x_axis.y = origin.y + Rotations[i].at<double>(3);
  		x_axis.z = origin.z + Rotations[i].at<double>(6);
  
  		y_axis.x = origin.x + Rotations[i].at<double>(1);
  		y_axis.y = origin.y + Rotations[i].at<double>(4);
  		y_axis.z = origin.z + Rotations[i].at<double>(7);

  		z_axis.x = origin.x + Rotations[i].at<double>(2);
  		z_axis.y = origin.y + Rotations[i].at<double>(5);
  	    z_axis.z = origin.z + Rotations[i].at<double>(8);  	

  		std::string x_axis_n = "x_axis" + std::to_string(i);
  		std::string y_axis_n = "y_axis" + std::to_string(i);
	    std::string z_axis_n = "z_axis" + std::to_string(i);

  		  //<,>: template
  		viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (origin, x_axis, 255.0, 0.0, 0.0, x_axis_n);
  		viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (origin, y_axis, 0.0, 255.0, 0.0, y_axis_n);
  		viewer->addLine<pcl::PointXYZ, pcl::PointXYZ> (origin, z_axis, 0.0, 0.0, 255.0, z_axis_n);
  	}

  	return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> PointCloud_sfm::simpleVis (
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{

    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
  
    return (viewer);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud_sfm::GeneratePointCloud(
	vector<Point3f> Pnts3D, vector<Vec3b> PntsColor)
{

	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (
		  new pcl::PointCloud<pcl::PointXYZRGB>);

	  int num_pnts = PntsColor.size();
	
	  for(int i=0; i<num_pnts; i++)
	  {
		  uchar blue = PntsColor[i].val[0];
		  uchar green = PntsColor[i].val[1];
		  uchar red = PntsColor[i].val[2];

		  pcl::PointXYZRGB point;

		  point.x = Pnts3D[i].x;
      point.y = Pnts3D[i].y;
      point.z = Pnts3D[i].z;
      point.r = red;
      point.g = green;
      point.b = blue;

      point_cloud_ptr->points.push_back (point);
	  }

	  return point_cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloud_sfm::GeneratePointCloud(
    vector<Point3f> Pnts3D){

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr (
      new pcl::PointCloud<pcl::PointXYZ>);

    int num_pnts = Pnts3D.size();

    for(int i=0; i<num_pnts; i++)
    {
      pcl::PointXYZ point;

      point.x = Pnts3D[i].x;
      point.y = Pnts3D[i].y;
      point.z = Pnts3D[i].z;

      point_cloud_ptr->points.push_back (point);
    }

    return point_cloud_ptr;
}