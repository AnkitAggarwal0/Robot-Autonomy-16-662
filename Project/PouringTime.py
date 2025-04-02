import numpy as np
from WeighingScale import WeighingScale
import time
from frankapy import FrankaArm
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.utils import min_jerk, min_jerk_weight


def Pour(tarWeight, currWeight, PrePourPose):
# using a time-based discretized pouring loop 
    pour_angle = np.pi/2
    rotation = np.array([[np.cos(pour_angle), 0, np.sin(pour_angle)], 
                         [0, 1, 0],
                         [np.sin(pour_angle), 0, np.cos(pour_angle)]]) #TODO: Tune/confirm axis 
    translation = np.array([0, 0, -0.1]) # TODO: Tune
    
    while (currWeight < tarWeight):
        # print(f"Current weight: {currWeight} g")
        # print(f"Target weight: {tarWeight} g")
        
        pour_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
        pour_pose.rotation = rotation
        pour_pose.translation = translation

        fa.set_cartesian_pose(pour_pose)

        time.sleep(0.1)

    fa.go_to_joints(PrePourPose)  # Reset to initial pose



if __name__ == '__main__':
    fa = FrankaArm()
    scale = WeighingScale()

    PrePourPose = np.array([]) #TODO: Get pre-pour pose 
    fa.go_to_joints(PrePourPose)


    previous_weight = scale.weight_averaged()
    target_weight = 50    
    current_weight = scale.weight_averaged()

    Pour(target_weight + previous_weight, current_weight, PrePourPose)

    final_weight = scale.weight_averaged()
    print(f"Final weight poured: {final_weight - previous_weight} g")
    
    fa.reset_joints()
    # previous_weight = final_weight  # Update previous weight for next use
    