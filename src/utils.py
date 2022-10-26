import numpy as np


def calculate_returns(rewards, gamma):
    result = np.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        result[t] = rewards[t] + gamma*result[t+1]
    return result

class Discretizer:
    def __init__(
        self,
        min_points,
        max_points,
        buckets,
        dimensions
        ):

        self.min_points = np.array(min_points)
        self.max_points = np.array(max_points)
        self.buckets = np.array(buckets)

        self.range = self.max_points - self.min_points
        self.spacing = self.range / self.buckets

        self.dimensions = dimensions

        self.n_states = np.round(self.buckets).astype(int)
        self.row_n_states = [self.n_states[dim] for dim in self.dimensions[0]]
        self.col_n_states = [self.n_states[dim] for dim in self.dimensions[1]]

        self.N = np.prod(self.row_n_states)
        self.M = np.prod(self.col_n_states)

        self.row_offset = [int(np.prod(self.row_n_states[i + 1:])) for i in range(len(self.row_n_states))]
        self.col_offset = [int(np.prod(self.col_n_states[i + 1:])) for i in range(len(self.col_n_states))]

    def get_index(self, state):
        state = np.clip(state, a_min=self.min_points, a_max=self.max_points)
        scaling = (state - self.min_points) / self.range
        idx = np.round(scaling * (self.buckets - 1)).astype(int)

        row_idx = idx[:, self.dimensions[0]]
        col_idx = idx[:, self.dimensions[1]]

        row = np.sum(row_idx*self.row_offset, axis=1)
        col = np.sum(col_idx*self.col_offset, axis=1)

        return row, col
