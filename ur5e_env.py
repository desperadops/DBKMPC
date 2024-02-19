import pybullet as pb
import pybullet_data
import numpy as np

class UR5EEnv(object):

    def __init__(self, render=True, ts=0.002):

        self.frame_skip = 10
        if render:
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)
        pb.setTimeStep(ts)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        flags = pb.URDF_USE_SELF_COLLISION
        planeID = pb.loadURDF('plane.urdf')
        self.robot = pb.loadURDF("./ur_e_description/urdf/ur5e.urdf", [0., 0., 0.], useFixedBase=1, flags=flags)
        pb.setGravity(0,0,-9.81)
        # self.reset_joint_state = [0., -0.78, 0., -2.35, 0., 1.57, 0.78]
        self.reset_joint_state = [-0.4516, -1.8354, 2.0225, 2.9517, -1.1194, 0.0012]
        # self.reset_joint_state = [2.833037823643068, -1.7208878855915042, 1.9337627000111866, 4.499652899403558, -1.5708362498441697, 2.8330936216032394]# 画图中心
        self.ee_id = 7  # 末端执行器id
        # self.sat_val = 0.3  # 输入大小
        self.sat_val = 1
        self.reset()
        self.joint_low = np.array([-3.14,-3.14,-3.14,-3.14,-3.14,-3.14])
        self.joint_high = np.array([3.14,3.14,3.14,3.14,3.14,3.14])
        self.Nstates = 15
        self.udim = 6
        self.dt = self.frame_skip*ts

    def get_ik(self, position, orientation=None):
        if orientation is None:
            jnts = pb.calculateInverseKinematics(self.robot, self.ee_id, position)
        else:
            jnts = pb.calculateInverseKinematics(self.robot, self.ee_id, position, orientation)
        return jnts

    def get_state(self):
        jnt_st = pb.getJointStates(self.robot, range(1, 7))
        ## 1,2,3 position
        ## 4,5,6 are the link frame pose, orientation in quat and ve
        ee_state = pb.getLinkState(self.robot, self.ee_id)[-2:] ### Why local and not the Cartesian ones? Ask Ian
        jnt_ang = []
        jnt_vel = []
        for jnt in jnt_st:
            jnt_ang.append(jnt[0])
            jnt_vel.append(jnt[1])
        # print(ee_state[0],ee_state[1],jnt_ang,jnt_vel)
        self.state = np.concatenate([ee_state[0], ee_state[1], jnt_ang, jnt_vel]) # ee_state[0] are [x,y,z] of EE, ee_state[1] are quaternions: x,y,z,w of EE
        # self.state = np.concatenate([ee_state[0], jnt_ang, jnt_vel])  # 控制时 用这一行 训练时 用上一行
        # self.state = np.concatenate([ee_state[0], jnt_ang])
        return self.state.copy()

    def reset(self):
        for i, jnt in enumerate(self.reset_joint_state):
            pb.resetJointState(self.robot, i+1, self.reset_joint_state[i])
        return self.get_state()

    def reset_state(self,joint):
        for i, jnt in enumerate(joint):
            pb.resetJointState(self.robot, i+1, joint[i])
        return self.get_state()

    def step(self, action):
        a = np.clip(action, -self.sat_val, self.sat_val)
        pb.setJointMotorControlArray(
                    self.robot, range(1, 7),
                    pb.VELOCITY_CONTROL, targetVelocities=a)
        for _ in range(self.frame_skip):
            pb.stepSimulation()

        return self.get_state()


if __name__ == "__main__":
    env = UR5EEnv(render=True)
    print(env.robot)



