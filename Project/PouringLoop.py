import numpy as np
from WeighingScale import WeighingScale
import time
from frankapy import FrankaArm
from autolab_core import RigidTransform

def PourPitch(target_weight, currWeight):
    dt = 0.01
    control_step = 0.1

    error = target_weight - currWeight
    error_prev = 0 
    error_integral = 0
    error_derivative = 0

    K_p, K_d, K_i = 0.1, 0.1, 0.1
    
    # print(f"Current weight: {currWeight} g")
    error = (target_weight - currWeight)/target_weight #Normalised error 
    error_integral += error * dt
    error_derivative = (error - error_prev) / dt
    error_prev = error
    # print(f"Error: {error} g")

    control_signal = K_p * error + K_d * error_derivative + K_i * error_integral #pitch angle - check for positive or negative 
    control_signal = np.clip(control_signal, -control_step,control_step) 
    print(f"Control signal: {control_signal}")

    rotation = np.array([[1, 0, 0], [0, np.cos(control_signal), -np.sin(control_signal)], [0, np.sin(control_signal), np.cos(control_signal)]]) # check axis needed...
    return rotation


if __name__ == '__main__':
    fa = FrankaArm()
    scale = WeighingScale()
    target_weight = 100 
    current_weight = scale.weight_averaged()
    while(current_weight < target_weight):
        rotation = PourPitch(scale, fa, target_weight)
        time.sleep(0.1)
        
        current_weight = scale.weight_averaged()
    

