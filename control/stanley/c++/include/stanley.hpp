#include <Eigen/Core>

using Eigen::Vector3d;
typedef Pose Eigen::Vector3d;

double AngDiff(double end_angle, double start_angle) {
  double delta_angle = end_angle - start_angle;
  double abs_delta_angle = fabs(delta_angle);
  double abs_complete_angle = 2 * M_PI + abs_delta_angle;
  double diff_angle;
  if (abs_complete_angle < abs_delta_angle) {
    diff_angle = -1 * sign(delta_angle) * abs_complete_angle;
  } else {
    diff_angle = delta_angle;
  }
  return diff_angle;
}

class Stanley {
  Stanley(double k_gain, double wheel_base, double max_steer_angle) {
    k_gain_ = k_gain;
    wheel_base_ = wheel_base;
    max_steer_angle_ = max_steer_angle;
  }
  double COntroller(Pose target_pose[3], Pose current_pose[3],
                    double current_velocity) {
    double yaw = current_pose[2];
    Vector2d norm_vector = {cos(yaw + M_PI / 2.0), sin(yaw + M_PI / 2.0)};
    double lateral_error =
        norm_vector.transpose() * (terget_pose.head(2) - current_pose.head(2));
    double k_soft = 1.0;
    double head_error = AngDiff(target_pose[2], current_pose[2]);
    double delta = headingError +
                   atan2(k_gain_ * lateral_error, k_soft + current_velocity);
    double diff_angle;
    if (fabs(delta) > max_steer_angle_) {
      diff_angle = sign(delta) * max_steer_angle_;
    } else {
      diff_angle = delta;
    }
    return diff_angle;
  }

  Pose ConvertToFrontWheelBy(Pose rear_wheel) {
    Pose front_wheel;
    front = rear_wheel + Vector3d(wheel_base_ * cos(rear_wheel[2]),
                                  wheel_base_ * sin(rear_wheel[2]), 0);
    return front_wheel;
  }

 private:
  double k_gain_;
  double wheel_base_;
  double max_steer_angle_;
}