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

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;


namespace myslam 
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING_Set_Ref=-1,
	INITIALIZING_Get_Depth=0,
        OK,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    Frame::Ptr  ref_;       // reference frame 
    Frame::Ptr  curr_;      // current frame 
    
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    vector<cv::Point3f>     pts_3d_ref_;        // 3d points in reference frame 
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    vector<cv::KeyPoint>    keypoints_ref_;     //keypoints in reference frame
    Mat                     descriptors_curr_;  // descriptor in current frame 
    Mat                     descriptors_ref_;   // descriptor in reference frame 
    Mat                     InitR_;    //rotation matrix when initializing
    Mat                     InitT_;    //translation matrix when initializing
    vector<cv::DMatch>      feature_matches_;
    
    SE3 T_c_r_estimated_;  // the estimated pose of current frame 
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    
    // parameters 
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;      // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 
    
protected:  
    // inner operation 
    void extractKeyPoints();
    void computeDescriptors();
    void extractKeyPoints_ref();
    void computeDescriptors_ref(); 
    void featureMatching();
    void poseEstimationPnP(); 
    void setRef3DPoints();
    
    void addKeyFrame();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    void triangulation ();
    void pose_estimation_2d2d ();

};
}

#endif // VISUALODOMETRY_H
