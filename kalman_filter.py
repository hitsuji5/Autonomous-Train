import numpy as np

class KalmanFilter(object):
    def __init__(self, F, R_w, H, R_v):
        self.F = F
        self.R_w = R_w
        self.H = H
        self.R_v = R_v

    def predict(self, x, P):
        x_bar = self.F.dot(x)
        P_bar = self.F.dot(P).dot(self.F.T) + self.R_w
        return x_bar, P_bar

    def update(self, x, P, z, R_v=None):
        if R_v is None:
            R_v = self.R_v
        e = z - self.H.dot(x)
        S = R_v + self.H.dot(P).dot(self.H.T)
        K = np.linalg.solve(S.T, self.H.dot(P)).T
        x_hat = x + K.dot(e)
        P_hat = (np.eye(len(x)) - K.dot(self.H)).dot(P)
        return x_hat, P_hat

    def predict_update(self, x, P, z, R_v=None):
        x_bar, P_bar = self.predict(x, P)
        x_hat, P_hat = self.update(x_bar, P_bar, z, R_v)
        return x_hat, P_hat


class TrainSimulator(object):
    def __init__(self, sig=3., num_sensor=3, sig_outlier=100, p_outlier=0.01, L=1e4, amax=1., bmax=1., vmax=300/3.6):
        self.vmax = vmax
        self.amax = amax
        self.bmax = bmax
        self.L = L
        self.sig = sig
        self.num_sensor = num_sensor
        self.sig_outlier = sig_outlier
        self.p_outlier = p_outlier

        # Initial State
        self.x = 0.
        self.v = 0.
        self.a = 0.

    def brake_point(self, distance_margin=5.0):
        brake_time = self.v/self.bmax
        brake_distance = self.v*brake_time - self.bmax/2*brake_time**2
        return self.L - brake_distance - distance_margin

    def move(self, dt, jerk=1., velocity_margin=1.):
        if self.x > self.brake_point():
            if self.v < velocity_margin:
                self.a = min(self.a + jerk * dt, -jerk * dt) if self.v > 0 else 0
            elif self.a > -self.bmax:
                self.a = max(self.a - jerk * dt, -self.bmax)
        else:
            if self.v < self.vmax - velocity_margin:
                self.a = min(self.a + jerk * dt, self.amax)
            elif self.a > 0:
                self.a = max(self.a - jerk * dt, 0)

        self.x += self.v * dt + self.a * dt ** 2 / 2
        self.v = max(min(self.v + self.a * dt, self.vmax), 0)

        return np.array([self.x, self.v, self.a])

    def measure(self):
        sig = np.array([self.sig if np.random.random() > self.p_outlier
                        else self.sig_outlier
                        for _ in range(self.num_sensor)])
        z = self.x + np.random.normal(loc=0.0, scale=sig, size=self.num_sensor)
        return z


if __name__ == '__main__':
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    G = np.array([dt**2/2, dt])
    R_w = G*G[None, :]*2
    H = np.array([[1, 0], [1, 0], [1, 0]])
    R_v = np.eye(3)*3.0**2

    x = np.zeros(2)
    P = np.zeros((2,2))
    train = TrainSimulator()
    kalman = KalmanFilter(F, R_w, H, R_v)
    x_true = []
    x_pred = []

    for _ in range(100):
        x_true.append(train.move(dt))
        z = train.measure()
        x, P = kalman.update(x, P, z)
        x_pred.append(x)
        x, P = kalman.predict(x, P)




