from dataSimulator import *
from matplotlib import pyplot as plt

def TEST_ACC():
    imu = IMU_Simulator()
    noise_list = []
    bias_list = []
    for i in range(0,1000):
        noise_list.append(imu.get_acc_noise())
        bias_list.append(imu.update_acc_bias())
        # imu.update_acc_bias()
    #visualize
    plt.figure()
    plt.title('IMU acc TEST')
    plt.plot(range(0,1000), noise_list, label='noise')
    print(bias_list)
    plt.plot(range(0,1000), bias_list, label='bias')
    plt.legend()
    plt.show()

class Measurement:
    def __init__(self):
        self.velVec = []
        self.noise_std = 1.5
    
    def add_vel(self, vel):
        self.velVec.append(vel + np.random.normal(0, self.noise_std))
    
class State:
    def __init__(self):
        self.vec = np.matrix([[0.0], [0.0]])

    def getVec(self):
        return vec
    
    def setFromVec(self, vec):
        pos = vec[0]
        vel = vec[1]
        self.vec = np.matrix([[pos], [vel]])

    def get_pos(self):
        return self.vec[0][0,0]
    
    def get_vel(self):
        return self.vec[1][0,0]

    def set_pos(self, pos):
        self.vec[0] = pos

    def set_vel(self, vel):
        self.vec[1] = vel


class StateRecorder:
    def __init__(self):
        self.posVec = []
        self.velVec = []
        self.accVec = []
    def add_pos(self, pos):
        self.posVec.append(pos)
    def add_vel(self, vel):
        self.velVec.append(vel)
    def add_acc(self, acc):
        self.accVec.append(acc)

    def recordAll(self, pos, vel, acc):
        self.add_pos(pos)
        self.add_vel(vel)
        self.add_acc(acc)

if __name__ == "__main__":
    # TEST_ACC()
    imu = IMU_Simulator()
    state = State()
    sim_state = State()
    gt = StateRecorder()
    sim_data = StateRecorder()
    mea = Measurement()
    total_simulation_time = 100 #s
    dt = 1 #s
    total_step = int(total_simulation_time/dt)
    init_acc = 0.2
    imu.set_acc(init_acc)

    # GroundTruth Generation
    for i in range(0, total_step):
        acc_now = imu.get_acc()
        dv = acc_now * dt
        dp = state.get_vel() * dt + 0.5 * dv * dt
        state.set_vel(state.get_vel() + dv)
        state.set_pos(state.get_pos() + dp)
        mea.add_vel(state.get_vel())
        gt.recordAll(state.get_pos(), state.get_vel(), acc_now)

    # Noisy IMU data generation
    for i in range(0, total_step):
        acc_now = imu.get_acc() + imu.get_acc_noise() * np.sqrt(dt) + imu.update_acc_bias()
        dv = acc_now * dt
        dp = sim_state.get_vel() * dt + 0.5 * dv * dt
        sim_state.set_vel(sim_state.get_vel() + dv)
        sim_state.set_pos(sim_state.get_pos() + dp)
        sim_data.recordAll(sim_state.get_pos(), sim_state.get_vel(), acc_now)

    # Kalman Filter
    now_state = State()
    prev_state = State()
    pred_state = State()
    noise_cov = np.array([[1, 0.0], [0.0, 1]])
    kf = StateRecorder()
    H = np.matrix([[0, 1]])
    A = np.matrix([[1, dt],
                   [0, 1]])
    B = np.matrix([[0.5 * dt * dt],[dt]])
    for i in range(total_step):
        # Prediction
        acc_ = sim_data.accVec[i]
        pred_state.vec = A * prev_state.vec + B * acc_
        noise_cov = A * noise_cov * A.transpose() + np.identity(2) * 0.01
        # Update
        innov = mea.velVec[i] - H * pred_state.vec
        R = np.matrix([[0.1]])
        K = noise_cov * H.transpose() * np.linalg.inv(H * noise_cov * H.transpose() + R)
        now_state.vec = pred_state.vec + K * innov
        noise_cov = (np.identity(2) - K * H) * noise_cov
        prev_state.vec = now_state.vec
        kf.recordAll(now_state.get_pos(), now_state.get_vel(), acc_)

    #visualize
    plt.figure()
    # plt.title('Kalman Filter')

    # plt.plot(range(0, total_step), kf.posVec, label='kf')
    # plt.plot(range(0, total_step), gt.posVec, label='gt', linestyle='dashed')
    # plt.plot(range(0, total_step), sim_data.posVec, label='sim')
    # # plt.plot(range(0, total_step), mea.velVec, label='mea')
    
    # plt.legend()
    # plt.show()

    # show in subplot
    plt.subplot(2,1,1)
    plt.title('Position')
    plt.plot(range(0, total_step), kf.posVec, label='kf')
    plt.plot(range(0, total_step), gt.posVec, label='gt', linestyle='dashed')
    plt.plot(range(0, total_step), sim_data.posVec, label='sim')
    plt.legend()

    plt.subplot(2,1,2)
    plt.title('Velocity')
    plt.plot(range(0, total_step), kf.velVec, label='kf')
    plt.plot(range(0, total_step), gt.velVec, label='gt')
    plt.plot(range(0, total_step), sim_data.velVec, label='sim')
    plt.plot(range(0, total_step), mea.velVec, label='mea')
    plt.legend()
    plt.show()


