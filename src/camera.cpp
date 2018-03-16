#include "myslam/camera.h"

namespace myslam
{

Camera::Camera()
{
}

Vector3d Camera::world2camera(const Vector3d & p_w, const SE3 & T_c_w)
{
	return T_c_w*p_w;
}
Vector3d Camera::camera2world(const Vector3d & p_c, const SE3 & T_c_w);
Vector3d Camera::camera2pixel(const Vector3d & p_c);
Vector3d Camera::pixel2camera(const Vector2d & p_p, double depth=1);
Vector3d Camera::pixel2world(const Vector2d & p_p, const SE3 & T_c_w, double depth=1 );
Vector3d Camera::world2pixel(const Vector3d & P_w, const SE3 & T_c_w);

}
