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
        // extract features from first frame 
        extractKeyPoints_ref();
	cout << "number of keypoints in the first frame" <<keypoints_ref_.size() <<endl;
        computeDescriptors_ref();
	cout << "number of descriptors in the first frame" << descriptors_ref_.size() << endl;
        break;
    }
    case INITIALIZING_Get_Depth:
    {
        state_ = OK;
        curr_ = frame;
        //extract features from second frame
        extractKeyPoints();
	cout << "number of keypoints in the second frame" <<keypoints_curr_.size() <<endl;
        computeDescriptors();
	cout << "number of descriptors in the second frame" << descriptors_curr_.size() << endl;
        featureMatching(); 
        pose_estimation_2d2d ();
        triangulation ();
	cout<<"number of 3d points is"<<pts_3d_ref_.size()<<endl;
        ref_ = curr_;
	descriptors_ref_ = Mat();
        for(cv::DMatch m: feature_matches_)
	{
	  descriptors_ref_.push_back( descriptors_curr_.row(m.queryIdx) );
	}  
        cout << "number of descriptors after triangulation is "<< descriptors_ref_.size();
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
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
    // select the features with depth measurements 
    descriptors_ref_ = Mat();
    Mat R = ( cv::Mat_ <double>(3,3)<<
        T_c_r_estimated_.rotation_matrix()(0,1) , T_c_r_estimated_.rotation_matrix()(0,2), T_c_r_estimated_.rotation_matrix()(0,3),
	T_c_r_estimated_.rotation_matrix()(1,1) , T_c_r_estimated_.rotation_matrix()(1,2), T_c_r_estimated_.rotation_matrix()(1,3),
	T_c_r_estimated_.rotation_matrix()(2,1) , T_c_r_estimated_.rotation_matrix()(2,2), T_c_r_estimated_.rotation_matrix()(2,3) 
    );
    
    Vector3d T;
    T << T_c_r_estimated_.translation()(0),T_c_r_estimated_.translation()(1), T_c_r_estimated_.translation()(2);
    for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    {
        
            descriptors_ref_.push_back(descriptors_curr_.row(i));
    }
    
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    for ( cv::DMatch m:feature_matches_ )
    {
        pts3d.push_back( pts_3d_ref_[m.queryIdx] );
        pts2d.push_back( keypoints_curr_[m.trainIdx].pt );
    }
    
    Mat K = ( cv::Mat_<double>(3,3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
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
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
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
    for ( DMatch m:feature_matches_ )
    {
        pts_1.push_back ( ref_->camera_->pixel2cam2d( keypoints_ref_[m.queryIdx].pt, K) );
        pts_2.push_back ( curr_->camera_->pixel2cam2d( keypoints_curr_[m.trainIdx].pt, K) );
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
    for(int i = 0; i<feature_matches_.size(); i++)
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

  
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) feature_matches_.size(); i++ )
    {
        points1.push_back ( keypoints_curr_[feature_matches_[i].queryIdx].pt );
        points2.push_back ( keypoints_ref_[feature_matches_[i].trainIdx].pt );
    }


    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( ref_->camera_->cx_, ref_->camera_->cy_ );
    int focal_length = ref_->camera_->fx_;
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, InitR_, InitT_, focal_length, principal_point );
    cout<<"R is "<<endl<<InitR_<<endl;
    cout<<"t is "<<endl<<InitT_<<endl;
}


}
