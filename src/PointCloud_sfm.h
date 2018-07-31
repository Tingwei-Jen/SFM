#ifndef PointCloud_sfm_H
#define PointCloud_sfm_H

#include <string>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

class PointCloud_sfm{
public:
	PointCloud_sfm();

	boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (
		pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
		vector<Mat> Rotations, vector<Mat> Translations);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (
		pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr GeneratePointCloud(
		vector<Point3f> Pnts3D, vector<Vec3b> PntsColor);

	pcl::PointCloud<pcl::PointXYZ>::Ptr GeneratePointCloud(
		vector<Point3f> Pnts3D);
private:

};
#endif