#ifndef PUREPURSUIT_H_
#define PUREPURSUIT_H_

class PurePursuit {
 public:
  PurePursuit(double k_gain, double wheel_base, double max_steer_angle) {
    k_gain_ = k_gain;
    wheel_base_ = wheel_base;
    max_steer_angle_ = max_steer_angle;
  }

  double Controller(double target_point[2], double current_pose[3],
                    double current_velocity) {
    double alpha = atan2(target_point[1] - current_pose[1],
                         target_point[0] - current_pose[0]) -
                   current_pose[2];
    double lookhead_distance = k_gain_ * current_velocity;
    double k_soft = 1.0;
    double delta =
        atan2(2 * wheel_base * sin(alpha), k_soft + lookhead_distance);
    double diff_angle;
    if (fabs(delat > max_steer_angle_)) {
      diff_angle = sign(delta) * max_steer_angle_;
    } else {
      diff_angle = delta;
    }
    return diff_angle;
  }

 private:
  double k_gain_;
  double wheel_base_;
  double max_steer_angle_;
}

#endif