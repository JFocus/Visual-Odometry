/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <boost/timer.hpp>
#include <sophus/se3.h>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/camera.h"
using namespace cv;

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING_Set_Ref ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING_Set_Ref:
    {
        state_ = INITIALIZING_Get_Depth;
        ref_ = frame;
        map_->insertKeyFrame ( frame );
	//detect keypoints in the first frame
	OP_extractKeyPoints();
        cout << "number of keypoints detected is" << OP_Keypoints_.size() << endl;
        break;
    }
    case INITIALIZING_Get_Depth:
    {
        state_ = OK; 
        curr_ = frame;
	OP_LKCal_Init();
        pose_estimation_2d2d ();
        triangulation ();
	cout<<"number of 3d points is"<<pts_3d_ref_.size()<<endl;
        ref_ = curr_;
	
	OP_Keypoints_ref_.clear();
	for (auto kp:OP_Keypoints_)
	  OP_Keypoints_ref_.push_back(kp);
	
	break;
    }
    case OK:
    {
        curr_ = frame;
	OP_LKCal();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
	    OP_Keypoints_ref_.clear();
	    for (auto kp:OP_Keypoints_)
	      OP_Keypoints_ref_.push_back(kp);
            setRef3DPoints();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    orb_->detect ( curr_->color_, keypoints_curr_ );
}

void VisualOdometry::extractKeyPoints_ref()
{
    orb_->detect ( ref_->color_, keypoints_ref_ );
}

void VisualOdometry::computeDescriptors()
{
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
}

void VisualOdometry::computeDescriptors_ref()
{
    orb_->compute ( ref_->color_, keypoints_ref_, descriptors_ref_ );
}

void VisualOdometry::featureMatching()
{
    // match desp_ref and desp_curr, use OpenCV's brute force match 
    static int i = 0;
    string name;
    Mat img_goodmatches;
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher ( cv::NORM_HAMMING );
    matcher.match ( descriptors_ref_, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;
    //cout << "minimum distance is" << min_dis <<endl;
    feature_matches_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            feature_matches_.push_back(m);
        }
    }
    drawMatches(ref_->color_, keypoints_ref_, curr_->color_, keypoints_curr_, feature_matches_, img_goodmatches);
    //imshow("good matches", img_goodmatches);waitKey(0);
    name = "good_matches" + std::to_string(i) + ".png";
    imwrite(name, img_goodmatches);    
    cout<<"good matches: "<<feature_matches_.size()<<endl;
    i++;
}

void VisualOdometry::setRef3DPoints()
{
    vector<Point3d> points;
    Mat R = ( cv::Mat_ <double>(3,3)<<
        T_c_r_estimated_.rotation_matrix()(0,1) , T_c_r_estimated_.rotation_matrix()(0,2), T_c_r_estimated_.rotation_matrix()(0,3),
	T_c_r_estimated_.rotation_matrix()(1,1) , T_c_r_estimated_.rotation_matrix()(1,2), T_c_r_estimated_.rotation_matrix()(1,3),
	T_c_r_estimated_.rotation_matrix()(2,1) , T_c_r_estimated_.rotation_matrix()(2,2), T_c_r_estimated_.rotation_matrix()(2,3) 
    );
    
    Mat T = (cv::Mat_<double>(3,1)<<
       T_c_r_estimated_.translation()(0),T_c_r_estimated_.translation()(1), T_c_r_estimated_.translation()(2)
    );
    
    for ( size_t i=0; i<OP_Keypoints_.size(); i++ )
    {
        Mat pt3_trans = R * (Mat_<double>(3,1) << pts_3d_ref_[i].x , pts_3d_ref_[i].y , pts_3d_ref_[i].z) + T;
	Point3d p3d (
	  pt3_trans.at<double>(0,0),
          pt3_trans.at<double>(1,0),
          pt3_trans.at<double>(2,0)
	);
	points.push_back(p3d);
    }
    pts_3d_ref_.clear();
    for (int i = 0; i< pts_3d_ref_.size() ; i++)
    {
      pts_3d_ref_.push_back(points[i]);
    }
    
}

void VisualOdometry::poseEstimationPnP()
{
  
    Mat K = ( cv::Mat_<double>(3,3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_3d_ref_, OP_Keypoints_curr_, K, Mat(), rvec, tvec, false, 1000, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_r_estimated_ = SE3(
        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
//    if ( d.norm() > 5.0 )
  //  {
    //    cout<<"reject because motion is too large: "<<d.norm()<<endl;
      //  return false;
    //}
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame ( curr_ );
}

void VisualOdometry::triangulation ()
{
    vector<Point3f> points;
    pts_3d_ref_.clear();
    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        InitR_.at<double>(0,0), InitR_.at<double>(0,1), InitR_.at<double>(0,2), InitT_.at<double>(0,0),
        InitR_.at<double>(1,0), InitR_.at<double>(1,1), InitR_.at<double>(1,2), InitT_.at<double>(1,0),
        InitR_.at<double>(2,0), InitR_.at<double>(2,1), InitR_.at<double>(2,2), InitT_.at<double>(2,0)
    );
    
    Mat K = ( Mat_<double> ( 3,3 ) << 
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    vector<Point2f> pts_1, pts_2;
    for ( int i = 0; i < OP_Keypoints_.size(); i++ )
    {
        pts_1.push_back ( ref_->camera_->pixel2cam2d( OP_Keypoints_ref_[i], K) );
        pts_2.push_back ( curr_->camera_->pixel2cam2d( OP_Keypoints_curr_[i], K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0);
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
       points.push_back( p );
    }
    
    for(int i = 0; i<OP_Keypoints_.size(); i++)
    {
      Mat pt2_trans = InitR_ * (Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z) + InitT_;
      Point3f p3d (
	pt2_trans.at<float>(0,0),
        pt2_trans.at<float>(1,0),
        pt2_trans.at<float>(2,0)
      );
      pts_3d_ref_.push_back(p3d);
    }
}

void VisualOdometry::pose_estimation_2d2d ()
{
    
    Mat K = ( Mat_<double> ( 3,3 ) <<  
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1);



    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( OP_Keypoints_ref_, OP_Keypoints_curr_, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( ref_->camera_->cx_, ref_->camera_->cy_ );
    int focal_length = ref_->camera_->fx_;
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( OP_Keypoints_ref_, OP_Keypoints_curr_, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( OP_Keypoints_ref_, OP_Keypoints_curr_, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, OP_Keypoints_ref_, OP_Keypoints_curr_, InitR_, InitT_, focal_length, principal_point );
    cout<<"R is "<<endl<<InitR_<<endl;
    cout<<"t is "<<endl<<InitT_<<endl;
}

void VisualOdometry::OP_extractKeyPoints()
{
    vector<cv::KeyPoint> kps;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect( ref_->color_, kps );
    for ( auto kp:kps )
      OP_Keypoints_.push_back( kp.pt );  
    for ( auto kp:OP_Keypoints_)
      OP_Keypoints_ref_.push_back(kp);
}

void VisualOdometry::OP_LKCal_Init()
{
    static int imagindex = 0;
    string name;
    cv::calcOpticalFlowPyrLK(ref_->color_ , curr_->color_ , OP_Keypoints_ref_, OP_Keypoints_curr_, OP_status_, OP_error_);
    int i = 0;
    OP_Keypoints_ref_.clear();
    for( auto iter = OP_Keypoints_.begin(); iter != OP_Keypoints_.end() ; i++)
    {
      if(OP_status_[i] == 0)
      {
	iter = OP_Keypoints_.erase(iter);
	continue;
      }
      OP_Keypoints_ref_.push_back(*iter);
      *iter = OP_Keypoints_curr_[i];
      iter++;
    }
    
    OP_Keypoints_curr_.clear();
    for (auto kp:OP_Keypoints_)
      OP_Keypoints_curr_.push_back(kp);
    
    cout << "size of keypoint list is "<< OP_Keypoints_.size() << endl;
    cout << "size of reference points is" << OP_Keypoints_ref_.size() << endl;
    cout << "size of current points is" << OP_Keypoints_curr_.size() << endl;
    cv::Mat img_show = curr_->color_.clone();
    for (auto kp:OP_Keypoints_)
      cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0) , 1);
    name = "good_matches" + std::to_string(imagindex) + ".png";
    imwrite(name, img_show);   
    imagindex++;
}

void VisualOdometry::OP_LKCal()
{
    static int imagindex = 0;
    string name;
    vector<cv::Point3f>     pts_3d_ref_tmp;
    OP_Keypoints_curr_.clear();
    cv::calcOpticalFlowPyrLK(ref_->color_ , curr_->color_ , OP_Keypoints_ref_, OP_Keypoints_curr_, OP_status_, OP_error_);
    int i = 0;
    OP_Keypoints_ref_.clear();
    for( auto iter = OP_Keypoints_.begin(); iter != OP_Keypoints_.end() ; i++)
    {
      if(OP_status_[i] == 0)
      {
	iter = OP_Keypoints_.erase(iter);
	continue;
      }
      pts_3d_ref_tmp.push_back(pts_3d_ref_[i]);
      OP_Keypoints_ref_.push_back(*iter);
      *iter = OP_Keypoints_curr_[i];
      iter++;
    }
    
    pts_3d_ref_.clear();
    for (int i = 0; i < pts_3d_ref_tmp.size(); i++)
      pts_3d_ref_.push_back(pts_3d_ref_tmp[i]);
    
    OP_Keypoints_curr_.clear();
    for (auto kp:OP_Keypoints_)
      OP_Keypoints_curr_.push_back(kp);
    
    cout << "number of 3d points after LK is" << pts_3d_ref_.size()<< endl;
    cout << "size of keypoint list is "<< OP_Keypoints_.size() << endl;
    cout << "size of reference points is" << OP_Keypoints_ref_.size() << endl;
    cout << "size of current points is" << OP_Keypoints_curr_.size() << endl;
    cv::Mat img_show = curr_->color_.clone();
    for (auto kp:OP_Keypoints_)
      cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0) , 1);
    name = "good_matches_" + std::to_string(imagindex) + ".png";
    imwrite(name, img_show);   
    imagindex++;
}


}
