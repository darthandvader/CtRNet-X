#!/usr/bin/env python3
import numpy as np
from filterpy import monte_carlo
class ParticleFilter:
    def __init__(self, num_states, init_distribution, motion_model, obs_model, num_particles, min_num_effective_particles=1.5):
        self._particles = np.zeros((num_particles, num_states))
        self._weights = np.ones((num_particles))/float(num_particles)
        self._min_num_effective_particles = min_num_effective_particles

        self._init_distribution = init_distribution
        self._motion_model = motion_model
        self._obs_model = obs_model

        self._prev_joint_angles = None
        self._num_particles = num_particles

    def norm_weights(self):
        self._weights = self._weights/np.sum(self._weights)

    def init_filter(self, std):
        tiled_std = np.tile(std, (self._num_particles, 1))
        self._init_std = tiled_std
        self._particles = self._init_distribution(tiled_std)

    def predict(self, std):
        tiled_std = np.tile(std, (self._num_particles, 1))
        self._particles = self._motion_model(self._particles, tiled_std)


    def update(self, points_2d, ctrnet, joint_angles, cam, cTr, gamma):
        obs_probs = self._obs_model(self._particles, points_2d, ctrnet, joint_angles, cam, cTr, gamma)
        self._weights = self._weights*obs_probs[:, 0]
        min_weight = 1e-50
        self._weights = np.maximum(self._weights, min_weight)
        sum_weights = np.sum(self._weights)
        # print(sum_weights)
        # if sum_weights < 1e-30:
        #     print("injected rand particles")
        self.norm_weights()
        # for p_idx, particle in enumerate(self._particles):
        #      obs_prob = self._obs_model(particle, points_2d, ctrnet, joint_angles, cam, cTr, gamma)
        #      self._weights[p_idx] = self._weights[p_idx]*obs_prob
        # print(self._weights)
        # self.norm_weights()

        # TODO: resample if particles are depleted
        thresh = 1./np.sum(self._weights**2)
        # print(thresh)
        # print(thresh)
        # performance = np.sum(self._weights/num_particles)
        # print(performance)
        # print(np.max(self._weights))
        if self._prev_joint_angles is not None:
            did_not_move = np.any(np.isclose(self._prev_joint_angles, joint_angles))
            if self._min_num_effective_particles > thresh or np.isnan(thresh) or thresh > 4000:
                print("resampling")
                indices = monte_carlo.systematic_resample(self._weights)
                print(indices)
                self._particles = self._particles[indices, :]
                self._weights = np.ones((self._num_particles))/float(self._num_particles)

        self._prev_joint_angles = joint_angles

    def inject_random_particles(self, num_rep):
        std = np.array([
                1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, # ori
                1.0e-1, 1.0e-1, 1.0e-1, # pos
            ])
        rand_indices = np.random.randint(0, self._num_particles, size=num_rep)
        prob = np.sum(self._weights[rand_indices])
        leftover_prob = 1 - prob
        tiled_std = np.tile(std, (num_rep, 1))
        self._particles[rand_indices, :] = self._init_distribution(tiled_std)
        self._weights[rand_indices] = np.tile(leftover_prob, (num_rep, 1)).squeeze()/float(num_rep)

    def get_mean_particle(self):
        self.norm_weights()
        return np.dot(self._weights, self._particles)