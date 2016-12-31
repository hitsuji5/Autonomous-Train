import numpy as np
import heapdict as hd

class Operator(object):
    """Operator class generates the fastest movement path for a train
    in the state space (position, speed)
    """
    def __init__(self, dt, L=1e4, accel=1., brake=1.):
        self.dt = dt
        self.L = L
        self.accel = accel
        self.brake = brake

    def shortest_path(self, start=(0, 0), vmax=300/3.6):
        """A* search algorithm
        """
        Fail = []
        explored = set()
        cost = {}
        cost[start] = 0
        frontier = hd.heapdict(cost)
        path = {}
        while frontier:
            pstate, pcost = frontier.popitem()
            if self.is_goal(pstate):
                state = pstate
                run_curve = [state]
                while state != start:
                    action, state = path[state]
                    run_curve.append(state)
                return run_curve[::-1]

            explored.add(pstate)
            for (action, state) in self.train_dynamics(pstate, vmax):
                if state not in explored:
                    total_cost = pcost + self.dt - self.heuristic(pstate) + self.heuristic(state)
                    if state not in cost or cost[state] > total_cost:
                        cost[state] = total_cost
                        frontier[state] = total_cost
                        path[state] = (action, pstate)
        return Fail

    def heuristic(self, state):
        """heuristic function for A*
        """
        r, v = state
        return self.L - r

    def train_dynamics(self, state, vmax):
        r, v = state
        r_ = r + v * self.dt
        r_accel = r + v * self.dt + self.accel * self.dt ** 2
        r_brake = r + v * self.dt - self.brake * self.dt ** 2
        v_ = v
        v_accel = v + self.accel * self.dt
        v_brake = v - self.brake * self.dt
        s_ = (r_, v_)
        s_accel = (r_accel, v_accel)
        s_brake = (r_brake, v_brake)

        if r > self.L: return []
        elif r > self.brake_point(v): return [(-self.brake, s_brake)]
        elif v_accel > vmax: return [(0, s_), (-self.brake, s_brake)]
        elif v_brake < 0: return [(self.accel, s_accel), (0, s_)]
        else: return [(self.accel, s_accel), (0, s_), (-self.brake, s_brake)]

    def brake_point(self, v, margin=3.0):
        if v < 5: return self.L
        brake_time = v / self.brake
        brake_distance = v * brake_time - self.brake / 2 * brake_time ** 2
        return self.L - brake_distance - margin

    def is_goal(self, state, margin=3.0):
        r, v = state
        return True if v < 0.1 and r > self.L - margin and r < self.L else False


class Train(object):
    """Train class simulates real train dynamics
    """
    def __init__(self, drive_noise=0.1, sensor_noise=3., num_sensor=3,
                 sensor_outlier=100, p_outlier=0.01, mass=1000, init_state=None):
        self.mass = mass
        self.drive_noise = drive_noise
        self.sensor_noise = sensor_noise
        self.num_sensor = num_sensor
        self.sensor_outlier = sensor_outlier
        self.p_outlier = p_outlier
        self.x = [[0, 0, 0]] if init_state is None else init_state
        self.energy = []
        self.t = 0

    def move(self, driving_force, dt, c=1):
        r, v, a = self.x[-1]
        a = (driving_force - c * v) / self.mass
        a = np.random.normal(a, self.drive_noise)
        r += v * dt + a * dt ** 2 / 2
        v = v + a * dt
        self.x.append([r, v, a])
        self.energy.append(driving_force * dt)
        self.t += dt
        return [r, v, a]

    def measure(self):
        sigma = np.array([self.sensor_noise if np.random.random() > self.p_outlier
                        else self.sensor_outlier
                        for _ in range(self.num_sensor)])
        z = np.random.normal(self.x[-1][0], sigma, size=self.num_sensor)
        return z

    def is_stopped(self, L=1):
        r, v, a = self.x[-1]
        return True if v < 0.01 and r > L else False

    def eval(self, x_pred):
        x_true = np.array(self.x)
        mse = ((x_pred - x_true) ** 2).mean(0)
        return mse

    def print_state(self):
        r, v, a = self.x[-1]
        print "Time: %.1f (s), Position: %.1f (m), Speed: %.1f (m/s)" %(self.t, r, v)

class Kalman(object):
    """Kalman class is a Kalman Filter algorithm for train tracking
    """
    def __init__(self, x0, P0, dt=0.1, drive_noise=0.1, sensor_noise=3.0, num_sensor=3, sensor_delay=0.01):
        self.F = np.array([[1, dt], [0, 1]])
        self.B = np.array([dt**2/2, dt])
        G = np.array([dt ** 2 / 2, dt])
        self.R_w = G * G[:, None] * drive_noise ** 2
        self.H = np.array([[1, -sensor_delay] for _ in range(num_sensor)])
        self.sensor_noise = sensor_noise
        self.R_v = np.eye(num_sensor) * sensor_noise ** 2
        self.x = [x0]
        self.P = P0

    def update_noise(self, z):
        sigma_z = np.abs(z - np.median(z))
        sigma_z = np.maximum(sigma_z, max(self.sensor_noise, np.median(sigma_z)))
        self.R_v = np.diag(sigma_z ** 2)

    def predict(self, z, u=0):
        x = self.F.dot(self.x[-1]) + self.B * u
        P = self.F.dot(self.P).dot(self.F.T) + self.R_w
        e = z - self.H.dot(x)
        S = self.R_v + self.H.dot(P).dot(self.H.T)
        K = np.linalg.solve(S.T, self.H.dot(P)).T
        x_hat = x + K.dot(e)
        P_hat = (np.eye(len(x)) - K.dot(self.H)).dot(P)
        self.x.append(x_hat)
        self.P = P_hat
        return x_hat, P_hat


class Controller(object):
    """Controller class is a PID controller for train control
    """
    def __init__(self):
        self.Jp = 1
        self.Jd = 0
        self.Ji = 0
        self.run_curve = []

    def set_params(self, params):
        self.Jp = params[0]
        self.Jd = params[1]
        self.Ji = params[2]

    def set_run_curve(self, run_curve):
        self.run_curve = run_curve

    def target_speed(self, r, start_index=0):
        rp, vp = self.run_curve[start_index]
        for i, (r_, v_) in enumerate(self.run_curve[start_index+1:]):
            if r >= rp and r < r_:
                return (v_ - vp)/(r_ - rp)*(r - rp) + vp, i
            rp, vp = r_, v_
        return 0, start_index

    def run(self, train=None, filter=None, dt=0.1, amax=1.2, bmax=-1.2, verbose=False):
        if not train:
            train = Train()
        index = 0
        error = 0
        int_error = 0
        total_err = 0
        n = 0
        u = 0
        nmax = len(self.run_curve) * 2 / dt
        while not train.is_stopped() and n < nmax:
            if filter:
                z = train.measure()
                filter.update_noise(z)
                x, P = filter.predict(z, u)
                r, v = x
            else:
                r, v, _ = train.x[-1]

            target, index = self.target_speed(r + 1.0, index)
            diff_error = (target - v - error) / dt
            error = target - v
            int_error += error * dt
            u = self.Jp * error + self.Jd * diff_error + self.Ji * int_error
            u = min(max(u, bmax), amax)
            driving_force = train.mass * u
            if verbose:
                print driving_force
            train.move(driving_force, dt)
            total_err += (error ** 2)
            n += 1
        return train, total_err / float(n)

    def param_search(self, filter=None, tol=0.2):
        n_params = 3
        dparams = [1.0] * n_params
        params = np.random.random(n_params)
        self.set_params(params)
        _, best_error = self.run(filter=filter)
        while sum(dparams) > tol:
            for i in range(n_params):
                params[i] += dparams[i]
                self.set_params(params)
                _, err = self.run(filter=filter)
                if err < best_error:
                    best_error = err
                    dparams[i] *= 1.1
                else:
                    params[i] -= 2.0 * dparams[i]
                    self.set_params(params)
                    _, err = self.run(filter=filter)
                    if err < best_error:
                        best_error = err
                        dparams[i] *= 1.1
                    else:
                        params[i] += dparams[i]
                        dparams[i] *= 0.9
        self.set_params(params)
        return params

if __name__ == '__main__':
    operator = Operator(dt=1, L=10000)
    run_curve = operator.shortest_path()
    kalman = Kalman(x0=np.zeros(2), P0=np.zeros((2,2)))
    controller = Controller()
    controller.set_params([1, 0, 0])
    controller.set_run_curve(run_curve)
    train, err = controller.run(filter=kalman)
    print train.t, err
