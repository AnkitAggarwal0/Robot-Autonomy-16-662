import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.ForwardKin([0., 0., 0., 0., 0., 0., 0.])

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        self.q[0:-1] = ang

        # For base link 
        self.Tjoint[0] = rt.rpyxyz2H([0, 0, self.q[0]], [0, 0, 0])
        self.Tcurr[0] = self.Tlink[0] @ self.Tjoint[0]

        # For all other links, needed to compute base link separately to create all HTMs wrt to the base link
        for i in range(1, len(self.Rdesc)):
            self.Tjoint[i] = rt.rpyxyz2H([0, 0, self.q[i]], [0, 0, 0])
            self.Tcurr[i] = np.matmul(self.Tcurr[i-1], np.matmul(self.Tlink[i], self.Tjoint[i]))

        # Compute Jacobian
        for i in range(len(self.J)):
            axis = self.Tcurr[i][:3,2]
            pos = self.Tcurr[-1][:3,3] - self.Tcurr[i][:3,3]
            self.J[:3, i] = np.cross(axis, pos)
            self.J[3:, i] = axis


        return self.Tcurr, self.J

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''
        C = np.eye(6) * 1000
        C[0,0], C[1,1], C[2,2] = 1000000, 1000000, 1000000
        C_inv = np.linalg.inv(C)
        
        W = np.eye(7) 
        W[2,2], W[3,3], W[-1,-1], W[-1,0] = 100, 100, 100, 1
        print('W:' , W)
        W_inv = np.linalg.inv(W)

        pos_step_max = 0.01
        rot_step_max = 0.01
        
        iteration_count = 0
        pos_err_norm = rot_error_vector_norm = np.inf
        max_iterations = 2000

        TGoal_rot = TGoal[:3, :3]
        TGoal_pos = TGoal[:3, 3]
    
        def convergence(pos_err_norm, rot_error_vector_norm):
            return pos_err_norm < x_eps and rot_error_vector_norm < r_eps

        while not convergence(pos_err_norm, rot_error_vector_norm): #and iteration_count<max_iterations:
            # print('Iteration:', iteration_count)
            Hcurr, J = self.ForwardKin(ang)
            ee_rot = Hcurr[-1][:3, :3]
            ee_pos = Hcurr[-1][:3, 3]

            # rotation error
            rot_err = TGoal_rot @ ee_rot.T
            err_axis, theta =  rt.R2axisang(rot_err)
            rot_error_vector = np.array(err_axis) * theta
            
            # position error
            pos_err = np.array(TGoal_pos - ee_pos)
            
            # error 
            Err = np.concatenate((pos_err, rot_error_vector))
            if np.linalg.norm(Err) > pos_step_max:
                Err = (Err/np.linalg.norm(Err)) * pos_step_max

            #for convergence condition 
            pos_err_norm = np.linalg.norm(Err[:3])
            rot_error_vector_norm = np.linalg.norm(Err[3:])

            J_hash = W_inv @ J.T @ np.linalg.inv(J @ W_inv @ J.T + C_inv)
            delta_q = J_hash @ Err
            ang += delta_q
            
            iteration_count += 1
        
        print('Iterations for Convergence:', iteration_count)
        return ang, Err
        
    