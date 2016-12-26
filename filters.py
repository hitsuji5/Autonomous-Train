import numpy as np
from scipy import stats


class TrainSimulator(object):
    def __init__(self, sig=3., num_sensor=3, sig_outlier=100, p_outlier=0.01,
                 L=1e4, amax=1., bmax=1., vmax=300/3.6, init_state=None):
        self.vmax = vmax
        self.amax = amax
        self.bmax = bmax
        self.L = L
        self.sig = sig
        self.num_sensor = num_sensor
        self.sig_outlier = sig_outlier
        self.p_outlier = p_outlier
        self.x = [[0, 0, 0]] if init_state is None else init_state
        self.t = 0

    def brake_point(self, v, distance_margin=50.0):
        brake_time = v/self.bmax
        brake_distance = v*brake_time - self.bmax/2*brake_time**2
        return self.L - brake_distance - distance_margin

    def move(self, dt, jerk=1., velocity_margin=1.0):
        r, v, a = self.x[-1]
        if r > self.brake_point(v):
            if v < velocity_margin:
                a = min(a + jerk * dt, -jerk * dt) if v > 0 else 0
            elif a > -self.bmax:
                a = max(a - jerk * dt, -self.bmax)
        else:
            if v < self.vmax - velocity_margin:
                a = min(a + jerk * dt, self.amax)
            elif a > 0:
                a = max(a - jerk * dt, 0)

        r += v * dt + a * dt ** 2 / 2
        v = max(min(v + a * dt, self.vmax), 0)
        self.x.append([r, v, a])
        self.t += dt

        return [r, v, a]

    def measure(self):
        sig = np.array([self.sig if np.random.random() > self.p_outlier
                        else self.sig_outlier
                        for _ in range(self.num_sensor)])
        z = self.x[-1][0] + np.random.normal(loc=0.0, scale=sig, size=self.num_sensor)
        return z

    def is_stopped(self):
        r, v, a = self.x[-1]
        return True if v < 0.01 and r > self.L - 50.0 else False

    def eval(self, x_pred):
        x_true = np.array(self.x)
        mse = ((x_pred - x_true) ** 2).mean(0)
        return mse

    def print_state(self):
        r, v, a = self.x[-1]
        print "Time: %.1f (s), Position: %.1f (m), Speed: %.1f (m/s)" %(self.t, r, v)

class KalmanFilter(object):
    def __init__(self, F, R_w, H, R_v):
        self.F = F
        self.R_w = R_w
        self.H = H
        self.R_v = R_v

    def init_state(self, x, P):
        self.x = [x]
        self.P = P

    def set_noise(self, R):
        self.R_v = R

    def predict_update(self, z):
        # Predict
        x = self.F.dot(self.x[-1])
        P = self.F.dot(self.P).dot(self.F.T) + self.R_w

        # Update
        e = z - self.H.dot(x)
        S = self.R_v + self.H.dot(P).dot(self.H.T)
        K = np.linalg.solve(S.T, self.H.dot(P)).T
        x_hat = x + K.dot(e)
        P_hat = (np.eye(len(x)) - K.dot(self.H)).dot(P)
        self.x.append(x_hat)
        self.P = P_hat
        return x_hat, P_hat


class ParticleFilter(object):
    def __init__(self, F, R_w, H, R_v):
        self.F = F
        self.R_w = R_w
        self.H = H
        self.R_v = R_v

    def init_state(self, x, N):
        self.x = [x]
        self.p = np.array([x for _ in range(N)])

    def set_noise(self, R):
        self.R_v = R

    def resample(self, z):
        N = self.p.shape[0]
        predictions = self.p.dot(self.F.T) + np.random.multivariate_normal([0,0,0], self.R_w+np.eye(3)*1e-12, N)
        w = stats.multivariate_normal.pdf(z - predictions.dot(self.H.T), cov=self.R_v)
        index = int(np.random.random() * N)
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += np.random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            self.p[i] = predictions[index]
        self.x.append(np.median(self.p, axis=0))
        return self.p


if __name__ == '__main__':
    train = TrainSimulator()
    t = 0
    dt = 10
    while not train.is_stopped():
        print t, train.move(dt)
        t += dt





