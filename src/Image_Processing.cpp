#include <iostream>
#include "Image_Processing.h"

using namespace std;

#define PI 3.14159265

/*
public:
*/
Image_Processing::Image_Processing(ImgProConfig config_)
{
	cout<<"Construct ImageProcess!!"<<endl;
	this->K = config_.K;
    this->distCoeffs = config_.distCoeffs;
	this->SURFminHessian = config_.SURFminHessian;
	this->MatchminRatio = config_.MatchminRatio;

    math = new Math();
}

Image_Processing::~Image_Processing()
{


}

void Image_Processing::GetFeatureSURFAll(vector<Mat> imgs, 
                                     vector<vector<KeyPoint>>& keypoints_all, 
                                     vector<Mat>& descriptors_all, 
                                     vector<vector<Vec3b>>& colors_all)
{
    keypoints_all.clear();
    descriptors_all.clear();
    colors_all.clear();

    int num_frames = imgs.size();
    for(int i=0; i<num_frames; i++)
    {
        cout<<"Get Feature of Frame: "<<i<<endl;
        vector<KeyPoint> keypoints;
        Mat descriptors;
        vector<Vec3b> colors;
        GetFeatureSURF(imgs[i], keypoints, descriptors, colors);
        keypoints_all.push_back(keypoints);
        descriptors_all.push_back(descriptors);
        colors_all.push_back(colors);
    }
}

void Image_Processing::MatchFeatureAll(vector<Mat> descriptors_all,
                                   vector<vector<DMatch>>& matches_all)
{
    matches_all.clear();
    int num_matches = descriptors_all.size()-1;
    for(int i=0; i<num_matches; i++)
    {
        vector<DMatch> matches;
        MatchFeatures(descriptors_all[i],descriptors_all[i+1],matches);
        cout << "Matching Images " << i << " and " << i + 1 ;
        cout << " matches size : "<< matches.size() <<endl;
        matches_all.push_back(matches);
    }
}

void Image_Processing::Init_Construction(vector<KeyPoint> keypoints_first_frame, 
                                     vector<KeyPoint> keypoints_second_frame,
                                     vector<DMatch> matches,
                                     vector<Vec3b> colors_first_frame, 
                                     vector<Vec3b>& total_colors,
                                     vector<Point3f>& total_pnts3D,
                                     vector<vector<int>>& pnts3D_index,
                                     Mat& Rot_second_frame, 
                                     Mat& Trans_second_frame)
{

    GetMatchPointColor(colors_first_frame, matches, total_colors);

	vector<Point2f> pnts1, pnts2;
	GetMatchPointLocation(keypoints_first_frame, keypoints_second_frame, matches, pnts1, pnts2);

	Mat R,T, mask; 
	if(Find_Transform_RT(pnts1, pnts2, this->K, R, T, mask)){
    	cout<<"Find Rot Trans"<<endl;
    } else {
    	cout<<"Cannot find rot and trans"<<endl;
    }

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);


    //total_pnts3D = math->Triangulation(this->K, R0, T0, R, T, pnts1, pnts2);
    total_pnts3D = math->triangulation_opencv(pnts1, pnts2, K, R, T);
    //給 match到的 keypoints  index, 沒有match 到的 keppoints still = -1
    //same match has same index
    //same ordering of Structure_index and pnts3D
    for(int i=0; i<matches.size(); i++)
    {
      pnts3D_index[0][matches[i].queryIdx] = i;
      pnts3D_index[1][matches[i].trainIdx] = i;
    }

    Rot_second_frame = R;
    Trans_second_frame = T;
}

void Image_Processing::New_Construction(vector<KeyPoint> keypoints_last_frame, 
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
                                    Mat& Trans_new_frame)
{

    GetMatchPointColor(colors_last_frame, matches, new_colors);

    vector<Point2f> imgpnts;
    vector<Point3f> object3Dpnts;

    for(int i=0; i< matches.size();i++)
    {
        int queryIdx =  matches[i].queryIdx;   //2
        int trainIdx =  matches[i].trainIdx;   //3
        int idx = pnts3D_index_last_frame[queryIdx];
        
        //只有一部分kp in last frame 有 3d points 可以做 pnp 
        if ( idx != -1)   //that keypoint has been matched
        {
            object3Dpnts.push_back(total_pnts3D[idx]);
            imgpnts.push_back(keypoints_new_frame[trainIdx].pt);
        } 
    }
    Mat rvec, R, T;
    solvePnPRansac(object3Dpnts, imgpnts, this->K, this->distCoeffs, rvec, T);
    Rodrigues(rvec, R);

    vector<Point2f> pnts1, pnts2;
    GetMatchPointLocation(keypoints_last_frame, keypoints_new_frame, matches, pnts1, pnts2);

    new_pnts3D = math->Triangulation(this->K, Rot_last_frame, Trans_last_frame, R, T, pnts1, pnts2);
    Rot_new_frame = R;
    Trans_new_frame = T;
}

void Image_Processing::Fusion_Construction(vector<DMatch> matches,
                                       vector<Point3f> new_pnts3D,
                                       vector<Vec3b> new_colors,
                                       vector<int>& pnts3D_index_last_frame,
                                       vector<int>& pnts3D_index_new_frame,
                                       vector<Point3f>& total_pnts3D,
                                       vector<Vec3b>& total_colors)
{
    for(int i=0; i<matches.size();i++)
    {
        int queryIdx =  matches[i].queryIdx;
        int trainIdx =  matches[i].trainIdx;   
        int idx = pnts3D_index_last_frame[queryIdx];

        if (idx != -1)  //that keypoint has been matched
        {
            pnts3D_index_new_frame[trainIdx] = idx;   //give same idx to same match 
        } 
        else   //new keypoint match and new pnts3D, give new index
        {
            total_pnts3D.push_back(new_pnts3D[i]);
            total_colors.push_back(new_colors[i]);
            pnts3D_index_last_frame[queryIdx] = total_pnts3D.size()-1;
            pnts3D_index_new_frame[trainIdx] = total_pnts3D.size()-1;
        }
    }
}

//get correspondence of the KPs and 3D pnts of this frame
void Image_Processing::FindCorrespond3DandKP(vector<KeyPoint> keypoints_this_frame, 
                                         vector<Point3f> total_pnts3D, 
                                         vector<int> pnts3D_index,
                                         vector<Point3f>& pnts3D_this_frame,
                                         vector<Point2f>& pntsKP_to_3D)
{
    pnts3D_this_frame.clear();
    pntsKP_to_3D.clear();
    //check all KP find out which kp with pnts3D
    for(int i=0; i<pnts3D_index.size(); i++)
    {
        int idx = pnts3D_index[i];
        if( idx != -1 ) //this KP has pnt3D, record idx and i
        {
            pnts3D_this_frame.push_back( total_pnts3D[idx] );
            pntsKP_to_3D.push_back( keypoints_this_frame[i].pt );
        }
    }
}

void Image_Processing::ReprojectionError(vector<Point3f> pnts_3D,vector<Point2f> pnts_img,
                                     Mat Rotation, Mat Translation,
                                     vector<Point2f>& pnts_repro, double& error)
{
    pnts_repro.clear();
    //pnts_repro = math->reprojection(pnts_3D, this->K, Rotation, Translation);
    pnts_repro = reprojection_with_distort(pnts_3D, this->K, this->distCoeffs,
                                           Rotation, Translation);

    error = 0;

    int num_pnts = pnts_img.size();
    for(int i=0; i<num_pnts; i++)
    {
        double x_drift = pnts_img[i].x - pnts_repro[i].x;
        double y_drift = pnts_img[i].y - pnts_repro[i].y;

        error = error + x_drift*x_drift + y_drift*y_drift;
    }
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Point3f Image_Processing::rotationMatrixToEulerAngles(Mat &R)
{
 
    assert(isRotationMatrix(R));
     
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Point3f(x*180/PI, y*180/PI, z*180/PI);
}


/*
private:
*/
void Image_Processing::GetFeatureSURF(
	Mat img, vector<KeyPoint>& keypoints, Mat& descriptors, vector<Vec3b>& colors)
{
	Ptr<SURF> detector = SURF::create();
    //Ptr<SIFT> detector_SIFT = SIFT::create(17000);
  	detector->setHessianThreshold(this->SURFminHessian);
 	detector->detectAndCompute( img, Mat(), keypoints, descriptors );

 	colors.resize(keypoints.size());
    for (int i=0; i<keypoints.size(); i++)
    {
        Point2f& p = keypoints[i].pt;
        colors[i] = img.at<Vec3b>(p.y, p.x);          //vec3b ordering: bgr
    }
}

/*
knn match: Get k best matches
使用KNN-matching算法，令K=2。则每个match得到两个最接近的descriptor，
然后计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match。
KNN match will return 2 nearest matches for each query descriptor
*/
void Image_Processing::MatchFeatures(Mat query, Mat train, vector<DMatch>& matches)
{
	vector<vector<DMatch> > knn_matches; //2 dimension: due to get k best matches
    BFMatcher matcher(NORM_L2);

    //KnnMatch(queryDescriptors, trainDescriptors)
    matcher.knnMatch( query, train, knn_matches, 2); //k=2 

    //获取满足Ratio Test的 min_dist and max_dist
    float min_dist = 500;

	for(int r=0; r<knn_matches.size(); r++)
    {
        DMatch bestMatch = knn_matches[r][0];
        DMatch betterMatch = knn_matches[r][1];
        float distanceRatio = bestMatch.distance/betterMatch.distance; 

        //滿足條件後 update mindist
        if(distanceRatio < this->MatchminRatio)
        {
        	float dist = knn_matches[r][0].distance;
        	if(dist<min_dist) min_dist = dist;
        }
    }

    //排除不满足Ratio Test的点和匹配距离过大的点
    for(int r=0; r<knn_matches.size(); r++)
    {
        DMatch bestMatch = knn_matches[r][0];
        DMatch betterMatch = knn_matches[r][1];
        float distanceRatio = bestMatch.distance/betterMatch.distance;  

        if(distanceRatio<this->MatchminRatio && knn_matches[r][0].distance< 5*max(min_dist,10.0f)){
          matches.push_back(knn_matches[r][0]);
        }
    }
}

bool Image_Processing::Find_Transform_RT(vector<Point2f> p1, vector<Point2f> p2, Mat K,
					                  Mat& Rot, Mat& Trans, Mat& mask)
{

	//K : camera intrinsic matrix
    double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
    Point2d principle_point(K.at<double>(2), K.at<double>(5));

    Mat Essentail_Matrix 
    	= findEssentialMat( p1, p2, focal_length, principle_point, 
                            RANSAC, 0.999, 1.0, mask);

    if (Essentail_Matrix.empty())
    {
    	cout<<"Essentail_Matrix is empty"<<endl;
    	return false;
    } 
    	
    //mask
    //0: outliers, 1: the other points
    double feasible_count = countNonZero(mask);
    cout <<"findEssentialMat  feasible_count: "<<(int)feasible_count << " -in- " << p1.size() << endl;

    //number of outliers > 50%  would be undependable
    if (feasible_count <= 15 || ( feasible_count / p1.size() ) < 0.6)
    {
        cout<<"Outliers too much"<<endl;
        return false;
    }

    //Recover relative rotation and translation from an estimated matrix
    //Return the number of inliers which pass the check (make sure their depths are positive)
    int pass_count = recoverPose( Essentail_Matrix, p1, p2, 
                                  Rot, Trans, focal_length, principle_point, mask );

    //同时位于两个相机前方的点的数量要足够大
    if (((double)pass_count) / feasible_count < 0.7)
    {
        cout<<"Front points not enough"<<endl;
        return false;
    }

    return true;
}

void Image_Processing::GetMatchPointLocation(vector<KeyPoint> kp1,vector<KeyPoint> kp2, 
                           vector<DMatch> matches,
                           vector<Point2f>& pnt1, vector<Point2f>& pnt2)
{
    for( int i = 0; i < matches.size(); i++ ){
      //-- Get the keypoints pixel loc from the good matches
      pnt1.push_back( kp1[ matches[i].queryIdx ].pt );
      pnt2.push_back( kp2[ matches[i].trainIdx ].pt );
    }
}

void Image_Processing::GetMatchPointColor(vector<Vec3b> colors, vector<DMatch> matches,
                                      vector<Vec3b>& Outcolors_queryIdx)
{
    for( int i = 0; i < matches.size(); i++ )
    {
      //-- Get the keypoints from the matches
      Outcolors_queryIdx.push_back( colors[ matches[i].queryIdx ] );
    }
}

void Image_Processing::GetMatchPointColor(vector<Vec3b> colors1, vector<Vec3b> colors2, 
                        vector<DMatch> matches,
                        vector<Vec3b>& Outcolors1, vector<Vec3b>& Outcolors2)
{
    for( int i = 0; i < matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      Outcolors1.push_back( colors1[ matches[i].queryIdx ] );
      Outcolors2.push_back( colors2[ matches[i].trainIdx ] );
    }
}

// Checks if a matrix is a valid rotation matrix.
bool Image_Processing::isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
     
    return  norm(I, shouldBeIdentity) < 1e-6;
     
}

vector<Point2f> Image_Processing::reprojection_with_distort(
                            vector<Point3f> pnts3D_this_frame,  
                            Mat K, Mat distCoeffs, Mat Rotation, Mat Translation)
{
    Mat rvec;
    Rodrigues(Rotation, rvec);
    vector<Point2f> projectedPoints;
    projectPoints(pnts3D_this_frame, rvec, Translation, K, distCoeffs, projectedPoints);
    return projectedPoints;
}