import numpy as np
import gym
# Implementation by osannolik: https://github.com/osannolik/gym-goddard
# Modified by Sergio Rozada


class Rocket(object):

    '''
        Expected parameters and methods of the rocket environment being simulated.
        V0/H0/M0:   Initial velocity/height/weight
        M1:         Final weight (set equal to dry mass if all fuel should be used)
        THRUST_MAX: Maximum possible force of thrust [N]
        GAMMA:      Fuel consumption [kg/N/s]
        DT:         Assumed time [s] between calls to step()
    '''

    V0 = H0 = M0 = M1 = THRUST_MAX = GAMMA = DT = None

    H_MAX_RENDER = None # Sets upper window bound for rendering

    def drag(self, v, h):
        raise NotImplementedError

    def g(self, h):
        raise NotImplementedError

class Default(Rocket):

    '''
        Models the surface-to-air missile (SA-2 Guideline) described in
        http://dcsl.gatech.edu/papers/jgcd92.pdf
        https://www.mcs.anl.gov/~more/cops/bcops/rocket.html
        The equations of motion is made dimensionless by scaling and choosing
        the model parameters in terms of initial height, mass and gravity.
    '''

    DT = 0.001

    V0 = 0.0
    H0 = M0 = G0 = 1.0

    H_MAX_RENDER = 1.015

    HC = 500.0
    MC = 0.6
    VC = 620.0

    M1 = MC * M0

    thrust_to_weight_ratio = 3.5
    THRUST_MAX = thrust_to_weight_ratio * G0 * M0
    DC = 0.5 * VC * M0 / G0
    GAMMA = 1.0 / (0.5*np.sqrt(G0*H0))

    def drag(self, v, h):
        return self.DC * v * abs(v) * np.exp(-self.HC*(h-self.H0)/self.H0)

    def g(self, h):
        return self.G0 * (self.H0/h)**2

class CustomGoddardEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, rocket=Default()):
        super(CustomGoddardEnv, self).__init__()

        self._r = rocket
        self.viewer = None

        self.U_INDEX = 0
        self.action_space = gym.spaces.Box(
            low   = np.array([0.0]),
            high  = np.array([1.0]),
            shape = (1,),
            dtype = np.float
        )

        self.V_INDEX, self.H_INDEX, self.M_INDEX = 0, 1, 2
        self.observation_space = gym.spaces.Box(
            low   = np.array([np.finfo(np.float).min, 0.0, self._r.M1]),
            high  = np.array([np.finfo(np.float).max, np.finfo(np.float).max, self._r.M0]),
            dtype = np.float
        )

        self.reset()

    def extras_labels(self):
        return ['action', 'thrust', 'drag', 'gravity']

    def step(self, action):
        v, h, m = self._state

        is_tank_empty = (m <= self._r.M1)

        a = 0.0 if is_tank_empty else action[self.U_INDEX]
        thrust = self._r.THRUST_MAX*a

        self._thrust_last = thrust

        drag = self._r.drag(v,h)
        g = self._r.g(h)

        # Forward Euler
        self._state = (
            0.0 if h==self._r.H0 and v!=0.0 else (v + self._r.DT * ((thrust-drag)/m - g)),
            max(h + self._r.DT * v, self._r.H0),
            max(m - self._r.DT * self._r.GAMMA * thrust, self._r.M1)
        )

        self._h_max = max(self._h_max, self._state[self.H_INDEX])

        is_done = bool(
            is_tank_empty and self._state[self.V_INDEX] < 0 and self._h_max > self._r.H0
        )

        if is_done:
            reward = 100*(self._h_max - self._r.H0)
        else:
            reward = 0.0

        extras = dict(zip(self.extras_labels(), [action[self.U_INDEX], thrust, drag, g]))

        return self._observation(), reward, is_done, extras

    def maximum_altitude(self):
        return self._h_max

    def _observation(self):
        return np.array(self._state)

    def reset(self):
        self._state = (self._r.V0, self._r.H0, self._r.M0)
        self._h_max = self._r.H0
        self._thrust_last = None
        return self._observation()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
