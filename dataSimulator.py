import numpy as np


class IMU_Simulator:
    def __init__(self):
        self.acc = 0.0
        self.acc_bias = 0.0
        self.acc_noise_std = 0.5
        self.acc_bias_random_walk_std = 0.03


    def get_acc(self):
        return self.acc

    def set_acc(self, acc):
        self.acc = acc

    def update_acc_bias(self):
        self.acc_bias = self.acc_bias + np.random.normal(0, self.acc_bias_random_walk_std)
        return self.acc_bias

    def get_acc_noise(self):
        return np.random.normal(0, self.acc_noise_std)

    def get_acc_bias(self):
        return self.acc_bias

