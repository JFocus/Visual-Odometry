#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

//Eigen include
#include <Eigen/Core>
#include <Eigen/Geometry>

using Eigen::Vector2d;
using Eigen::Vector3d;

//Sophus include
#include <sophus/se3.h>
using Sophus::SE3;

//cv include

#include <opencv2/core/core.hpp>
using cv::Mat;

//std
#include <vector>
#include <list>
#include <memory>
#include <iostream>
#include <string>
#include <set>
#include <unordered_map>
#include <map>


using namespace std;


#endif
