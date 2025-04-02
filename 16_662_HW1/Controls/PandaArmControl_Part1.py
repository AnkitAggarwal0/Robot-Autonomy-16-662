import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion


# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque.xml"

################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):  # TODO:
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'hand')

    # Get the Jacobian for the desired location on the robot (The end-effector)
    jacp = np.zeros((3, 7))
    jacr = np.zeros((3, 7))
    mj.mj_jacBody(model, data, jacp, jacr, body_id)

    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    # Specify the desired force in global coordinates
    desired_force = np.array([15, 0, 0])

    # Compute the required control input using desied force values
    input_req = jacp.T @ desired_force

    # Set the control inputs
    data.ctrl[:7] = input_req + data.qfrc_bias[:7]
    
    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

# Control callback for an impedance controller
def impedance_control(model, data):  # TODO:

    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    body_hand = data.body("hand")
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'hand')

    # Set the desired position
    desired_ee_position = np.array([0.6, 0.0, 0.6])

    # Set the desired velocities
    desired_ee_velocity = np.array([0.0, 0.0, 0.0])

    # Set the desired orientation (Use numpy quaternion manipulation functions)
    desired_orientation = np.quaternion(-0.5, 0.5, -0.5, 0.5)

    # Get the current orientation
    current_orientation = np.quaternion(*data.xquat[body_id]) 

    # Get orientation error
    orientation_error = desired_orientation * current_orientation.inverse() 
    orientation_error = 2 * orientation_error.vec

    # Get the position error
    position_error = desired_ee_position - data.xpos[body_id] #body_hand.xpos
    velocity_error = desired_ee_velocity - data.cvel[body_id][3:] #body_hand.cvel[3:]

    # Get the Jacobian at the desired location on the robot
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    #full_jac = np.zeros((6, model.nv))
    
    mj.mj_jacBody(model, data, jacp, jacr, body_id)

    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    # Compute the impedance control input torques
    Kp = 1.5
    Ko = 1.5
    Kd = 0.5

    F_des = (Kp * position_error) + (Kd * velocity_error) + np.array([15.0, 0.0, 0.0])
    position_term = jacp.T @ F_des

    T_des = (Ko * orientation_error)
    orientation_term = jacr.T @ T_des
    
    impedance_control_input = position_term + orientation_term 

    # Set the control inputs
    data.ctrl[:7] = impedance_control_input[:7] + data.qfrc_bias[:7]
    
    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]
    # print(force[-1])


def position_control(model, data):
    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    desired_joint_positions = np.array([-1.381117, 0.50327976,  1.52171772, -1.88583765, -2.85264866,  2.71926329, -0.77118883])
    # print(desired_joint_positions)
    

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (np.array([0, 0, 0, 0, 0, 0, 0])-data.qvel[:7])


####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    # mj.set_mjcb_control(gravity_comp)  # TODO:
    # mj.set_mjcb_control(force_control)  
    # mj.set_mjcb_control(impedance_control)  
    mj.set_mjcb_control(position_control)  

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(
        force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    viewer.launch(model, data)
    # with viewer.launch_passive(model, data) as v:
    #     for _ in range(int(force_sensor_max_time/model.opt.timestep)):
    #         mj.mj_step(model, data)
    #         v.sync()
    #     input("Press anything to continue...") # omit this to autoclose the viewer when time is up

    # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt('force_vs_time_ic.csv', plot, delimiter=',')
