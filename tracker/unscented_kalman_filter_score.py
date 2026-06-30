"""
Instead of approximating non-function (like the EKF), the UKF approximates probability distribution. It uses a
principle called the Unscented Transform. It is easier to approximate a probability distribution than it is to
approximate an arbitrary nonlinear function.

Important steps:
1. Sigma Points: The UKF carefully selects a small, fixed set of sample points from the current state distribution
(mean and covariance). These are called sigma points.
2. Propagation: It passes each sigma point individually through the true nonlinear function (no linearization as in EKF).
3. Reconstruction: It calculates the weighted mean and covariance of the transformed points to get the new estimated
                   state.

Please look at the following for more details:
https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
About Coordinated Turn (CT) motion model, read on: https://ieeexplore.ieee.org/document/6916122
"""

import numpy as np
import scipy.linalg
from scipy.linalg import sqrtm

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of freedom (contains values for N=1, ..., 9). 
Taken from MATLAB/Octave's chi2inv function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

# Todo: These thresholds require tuning, particulary 'anglechange_threshold' and 'acceleration_threshold'.
# config = {'anglechange_threshold': 0.1, 'acceleration_threshold': 1.0, 'turnrate_threshold':0.01}
config = {'anglechange_threshold': 0.5, 'acceleration_threshold': 0.5, 'turnrate_threshold':0.01}


class UnscentedKalmanFilterScore(object):
    """
    Unscented Kalman Filter for tracking bounding boxes in image space.

    The 10-dimensional state space
        x, y, w, h, c, vx, vy, vw, vh, vc
    contains the bounding box center position (x, y), width w, height h, tracklet confidence (score) c and their
    respective velocities.

    Object motion follows a constant velocity model. The bounding box location and detection score, (x, y, w, h, c),
    are taken as direct observation of the state space (observation model).

    UKF parameters:
    - alpha: Determines spread of sigma points (typically 1e-3)
    - beta: Used to incorporate prior knowledge of distribution (beta=2 for Gaussian)
    - kappa: Secondary scaling parameter (usually k = 0, k = 3 - n, OR κ = 1)
    """

    def __init__(self, alpha=1e-3, beta=2.0, kappa=1.0):
        ndim, dt = 5, 1.

        # State space dimension, measurement space dimension, and time step
        self.ndim_state = 2 * ndim  # 10
        self.ndim_meas = ndim  # 5
        self.dt = dt  # time step

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha ** 2 * (self.ndim_state + kappa) - self.ndim_state  # Lambda - composite scaling parameter

        # Motion and observation uncertainty weights
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # Motion model (constant velocity)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation model (direct observation of the first 5 states)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Pre-compute weights for sigma points
        self._compute_weights()

        # Motion model parameters
        self.damping = 0.98  # Velocity damping
        self.max_acceleration = 50.0  # pixels/s²
        self.max_size_change = 0.3  # 30% per second

    def _compute_weights(self):
        """Compute scaling weights for sigma points"""
        n = self.ndim_state
        num_sigma = 2 * n + 1

        self.Wm = np.zeros(num_sigma)
        self.Wc = np.zeros(num_sigma)

        # Weights for mean and covariance
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, num_sigma):
            self.Wm[i] = 1 / (2 * (n + self.lambda_))
            self.Wc[i] = self.Wm[i]

    def _compute_sigma_points(self, mean, covariance):
        """Generate sigma points from mean and covariance"""
        n = self.ndim_state
        num_sigma = 2 * n + 1

        sigma_points = np.zeros((num_sigma, n))
        sigma_points[0] = mean

        # Matrix square root using Cholesky decomposition
        try:
            sqrt_P = sqrtm((n + self.lambda_) * covariance)
        except:
            # If covariance is not positive definite, add a small diagonal term
            covariance = covariance + np.eye(n) * 1e-6
            sqrt_P = sqrtm((n + self.lambda_) * covariance)

        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_P[i]
            sigma_points[n + i + 1] = mean - sqrt_P[i]

        return sigma_points

    def _unscented_transform(self, sigma_points, noise_cov=None):
        """Compute mean and covariance from sigma points"""
        n = self.ndim_state
        num_sigma = 2 * n + 1

        # Compute mean
        mean = np.zeros(n)
        # mean = np.zeros(sigma_points.shape[1])
        for i in range(num_sigma):
            mean += self.Wm[i] * sigma_points[i]

        # Compute covariance
        covariance = np.zeros((n, n))
        for i in range(num_sigma):
            y = sigma_points[i] - mean
            covariance += self.Wc[i] * np.outer(y, y)

        if noise_cov is not None:
            covariance += noise_cov

        return mean, covariance

    def _unscented_transform_z(self, sigma_points_z, noise_cov=None):
        """Compute predicted measurement mean and innovation covariance"""
        n = self.ndim_state
        num_sigma = 2 * n + 1

        # Compute predicted measurement mean
        z_pred = np.zeros(self.ndim_meas)
        for i in range(num_sigma):
            z_pred += self.Wm[i] * sigma_points_z[i]

        # Compute innovation covariance
        S = np.zeros((self.ndim_meas, self.ndim_meas))
        for i in range(num_sigma):
            res_z = sigma_points_z[i] - z_pred
            S += self.Wc[i] * np.outer(res_z, res_z)

        if noise_cov is not None:
            S += noise_cov

        return z_pred, S

    def _get_process_noise(self, mean):
        """Calculate process noise covariance matrix Q_k based on current state"""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4]]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[4]]

        return np.diag(np.square(np.r_[std_pos, std_vel]))

    def _get_measurement_noise(self, mean, det_score=None):
        """Calculate measurement noise covariance matrix R_k based on current state"""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            2 * self._std_weight_position * mean[4]]

        # Noise Scale Adaptive (NSA) for measurement noise
        if det_score is not None:
            std = [(1. - det_score) * x for x in std]

        return np.diag(np.square(std))

    # ======================
    def state_transition_amm(self, state, last_velocity):
        """
        Adaptive Motion Model (AMM)
        Adaptive state transition for multi-object tracking.
        Adaptively choose the best motion model i.e., switches between motion models based on motion characteristics or
        state feedback.
        """
        # Analyze motion patterns from history
        motion_type = self.classify_motion(state, last_velocity)

        if motion_type == 'constant_velocity':
            return self.state_transition_cv(state)
        elif motion_type == 'accelerating':
            return self.state_transition_ca_10d(state, last_velocity)
        elif motion_type == 'turning':
            return self.state_transition_ct_10d(state, last_velocity)

    def classify_motion(self, state, last_velocity):
        """
        Classify the motion type based on state history into Constant velocity (CV), Constant acceleration (CA) or
        Coordinated Turn (CT) model.
        """
        vx, vy = state[5], state[6]
        velocity_mag = np.sqrt(vx ** 2 + vy ** 2)  # Magnitude of a velocity vector, aka speed

        # # Detect turning  # Todo: uncomment this part to use CT with AMM but CT doesn't work properly at the moment.
        # if len(self.last_velocity) > 1:
        #     v_prev = last_velocity[-1][:2]
        #     angle_change = abs(np.arctan2(vy, vx) - np.arctan2(v_prev[1], v_prev[0]))  # Only vx and vy
        #     if angle_change >= config['anglechange_threshold']:  # Significant turn (in radian), angle change threshold
        #         print('TURNING: ----------------', angle_change)
        #         return 'turning'

        # Detect acceleration
        if len(last_velocity) > 1:
            acc = state[5:7] - last_velocity[-1][:2]  # Only acc_x and acc_y
            acc_magn = np.sqrt(acc[0] ** 2 + acc[1] ** 2)  # Magnitude of acceleration vector
            if abs(acc_magn) >= config['acceleration_threshold']:  # Significant acceleration, acceleration magnitude threshold
                print('ACCELERATING: ----------------', abs(acc_magn))
                return 'accelerating'

        return 'constant_velocity'

    def state_transition_cv(self, state):
        """
        Constant velocity (CV) model (10-D)
        For constant velocity, this is linear, but we implement as a function for consistency with UKF framework

        x_{k+1} = x_{k} + v_x{k} * dt
        y_{k+1} = y_{k} + v_y{k} * dt
        v_x{k+1} = v_x{k}
        v_y{k+1} = v_y{k}
        ...
        """
        state_new = state.copy()
        for i in range(5):
            state_new[i] = state[i] + state[i + 5] * self.dt
        return state_new # OR return np.dot(state, self._motion_mat.T)

    def state_transition_ca_10d(self, state, last_velocity):
        """
        Constant acceleration (CA) for 10-D state

        x_{k+1} = x_k + v_x{k}*dt + 0.5*a_x{k}*dt²
        y_{k+1} = y_k + v_y{k}*dt + 0.5*a_y{k}*dt²
        v_x{k+1} = v_x{k} + a_x{k}*dt
        v_y{k+1} = v_y{k} + a_y{k}*dt
        . . .
        """
        # Estimate acceleration from recent history
        if len(last_velocity) >= 1:
            v_prev = last_velocity[-1]
            v_curr = state[5:]
            acc = (v_curr - v_prev) / self.dt
        else:
            acc = np.array([0, 0, 0, 0, 0])  # For [ax, ay , aw, ah, ac]

        # print('ACCELERATING: ------ ', np.sqrt(acc[0] ** 2 + acc[1] ** 2))
        state_new = state.copy()
        for i in range(5):
            # Position: x + v*dt + 0.5*a*dt²
            state_new[i] = state[i] + state[i + 5] * self.dt + 0.5 * acc[i] * self.dt ** 2
            # Velocity: v + a*dt
            state_new[i + 5] = state[i + 5] + acc[i] * self.dt
        return state_new

    def state_transition_ct_10d(self, state, last_velocity):
        """
        Coordinated Turn (CT) Model with Cartesian Velocities for 10-D state.

        Exact solution for constant turn rate:
            x_{k+1} = x_k + (vx/ω)*sin(ωdt) - (vy/ω)*(1 - cos(ωdt))
            y_{k+1} = y_k + (vx/ω)*(1 - cos(ωdt)) + (vy/ω)*sin(ωdt)
            vx_{k+1} = vx*cos(ωdt) - vy*sin(ωdt)
            vy_{k+1} = vx*sin(ωdt) + vy*cos(ωdt)

        Estimate turn rate from heading changes.
        Δθ = θ_k - θ_k+1
        ω = Δθ/Δt where θ = atan2(vy, vx)

        """
        # Estimate Turn Rate (angular velocity) from Heading (angle) Changes
        vx, vy = state[5], state[6]
        v = np.sqrt(vx ** 2 + vy ** 2)  # Speed
        theta_curr = np.arctan2(vy, vx)  # Heading at a current time step, in radians
        if len(last_velocity) >= 1:
            v_prev = last_velocity[-1][:2]
            theta_prev = np.arctan2(v_prev[1], v_prev[0])  # Heading at a previous time step

            delta_theta = theta_curr - theta_prev
            # Compute angle difference with unwrapping. This ensures the angle difference is always in the range [−π, π].
            delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
            omega = delta_theta / self.dt  # Heading-based Turn rate ω (in radians per second)
        else:
            omega = 0.0

        x, y, w, h, c = state[:5]

        if abs(omega) < config['turnrate_threshold']:  # For position (when moving straight, omega ≈ 0)
            # Fallback to constant velocity
            x_new = x + vx * self.dt  # x_new = x + v * np.cos(theta_curr) * self.dt
            y_new = y + vy * self.dt  # y_new = y + v * np.sin(theta_curr) * self.dt
            vx_new = vx
            vy_new = vy
        else:  # For position (when turning, omega ≠ 0)
            # Coordinated turn with Cartesian velocity
            x_new = x + (vx / omega) * np.sin(omega * self.dt) - (vy / omega) * (1.0 - np.cos(omega * self.dt))
            y_new = y + (vx / omega) * (1.0 - np.cos(omega * self.dt)) + (vy / omega) * np.sin(omega * self.dt)
            vx_new = vx * np.cos(omega * self.dt) - vy * np.sin(omega * self.dt)
            vy_new = vx * np.sin(omega * self.dt) + vy * np.cos(omega * self.dt)


        # Width, height, confidence follow constant velocity
        w_new = w + state[7] * self.dt
        h_new = h + state[8] * self.dt
        c_new = c + state[9] * self.dt

        # return np.array([x_new, y_new, w_new, h_new, c_new,
        #                  state[5], state[6], state[7], state[8], state[9]])

        return np.array([x_new, y_new, w_new, h_new, c_new,
                         vx_new, vy_new, state[7], state[8], state[9]])

    # ======================

    def measurement_function(self, state):
        """
        Measurement function - direct observation of the first 5 states
        """
        return state[:5]  # The same as np.dot(self._update_mat, state)

    def initiate(self, measurement):
        """Create a track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h, c) with center position (x, y),
            width w, height h and tracklet confidence.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10 dimensional) of the new track.
            Unobserved velocities are initialized to 0.0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            20 * self._std_weight_position * measurement[4],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            100 * self._std_weight_velocity * measurement[4]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, velocity=()):
        """Run UKF prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.
        velocity : tuple
           Tuple of velocities from the last two time steps

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.

        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(mean, covariance)

        # Propagate sigma points through state transition
        sigma_points_pred = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            sigma_points_pred[i] = self.state_transition_amm(sp, velocity)  # This requires tuning of thresholds Todo:
            # sigma_points_pred[i] = self.state_transition_cv(sp)  # CV only
            # sigma_points_pred[i] = self.state_transition_ca_10d(sp, velocity)  # CA only
            # sigma_points_pred[i] = self.state_transition_ct_10d(sp, velocity)  # CT only  # Todo: why this doesn't work properly?

        # Get process noise
        Q = self._get_process_noise(mean)

        # Compute predicted mean and covariance using unscented transform
        mean_pred, covariance_pred = self._unscented_transform(sigma_points_pred, Q)

        return mean_pred, covariance_pred

    def project(self, mean, covariance, det_score=None):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (10 dimensional array).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        det_score : float
            Detection confidence score for Noise Scale Adaptive (NSA) filtering.

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(mean, covariance)

        # Propagate sigma points through measurement function
        sigma_points_z = np.zeros((2 * self.ndim_state + 1, self.ndim_meas))
        for i, sp in enumerate(sigma_points):
            sigma_points_z[i] = self.measurement_function(sp)

        # Compute projected mean and covariance using unscented transform
        mean_proj, covariance_proj = self._unscented_transform_z(sigma_points_z)

        # Add measurement noise
        R = self._get_measurement_noise(mean, det_score)
        covariance_proj += R

        return mean_proj, covariance_proj

    def multi_predict(self, mean, covariance, velocity=()):
        """Run UKF prediction step (Vectorized version).

        Parameters
        ----------
        mean : ndarray
            The Nx10 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx10x10 dimensional covariance matrices of the object states at the
            previous time step.
        velocity : tuple
           Tuple of velocities from the last two time steps

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        N = len(mean)
        mean_pred = np.zeros_like(mean)
        covariance_pred = np.zeros_like(covariance)

        for i in range(N):
            if len(velocity)!=0:
                mean_pred[i], covariance_pred[i] = self.predict(mean[i], covariance[i], velocity[i])
            else:
                mean_pred[i], covariance_pred[i] = self.predict(mean[i], covariance[i])

        return mean_pred, covariance_pred

    def update(self, mean, covariance, measurement, det_score=None):
        """Run UKF correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (x, y, w, h, c).
        det_score : float
            Detection confidence score for noise scale adaptive (NSA) filtering.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # Compute sigma points from a current state
        sigma_points = self._compute_sigma_points(mean, covariance)
        n = self.ndim_state
        num_sigma = 2 * n + 1

        # Transform sigma points through measurement function
        sigma_points_z = np.zeros((num_sigma, self.ndim_meas))
        for i, sp in enumerate(sigma_points):
            sigma_points_z[i] = self.measurement_function(sp)

        # Get measurement noise
        R = self._get_measurement_noise(mean, det_score)

        # Compute predicted measurement mean and innovation covariance
        z_pred, S = self._unscented_transform_z(sigma_points_z, R)

        # Compute cross-covariance from sigma points to compute gain
        Pxz = np.zeros((n, self.ndim_meas))
        for i in range(num_sigma):
            res_x = sigma_points[i] - mean
            res_z = sigma_points_z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(res_x, res_z)

        # Compute Kalman gain
        try:
            kalman_gain = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # If S is singular, use pseudo-inverse
            kalman_gain = Pxz @ np.linalg.pinv(S)

        # Compute innovation
        innovation = measurement - z_pred

        # Update state mean and covariance
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ S @ kalman_gain.T

        # Ensure covariance is symmetric positive definite
        new_covariance = (new_covariance + new_covariance.T) / 2

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (10 dimensional).
        covariance : ndarray
            Covariance of the state distribution (10x10 dimensional).
        measurements : ndarray
            An Nx5 dimensional matrix of N measurements, each in
            format (x, y, w, h, c).
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        metric : str
            Distance metric to use: 'maha' for Mahalanobis, 'gaussian' for Euclidean.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        # Project to measurement space using UKF projection
        mean_proj, covariance_proj = self.project(mean, covariance)

        if only_position:
            mean_proj = mean_proj[:2]
            covariance_proj = covariance_proj[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean_proj

        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            try:
                cholesky_factor = np.linalg.cholesky(covariance_proj)
                z = scipy.linalg.solve_triangular(
                    cholesky_factor, d.T, lower=True, check_finite=False,
                    overwrite_b=True)
                squared_maha = np.sum(z * z, axis=0)
                return squared_maha
            except np.linalg.LinAlgError:
                # If covariance is singular, use a regularized version
                covariance_proj += np.eye(len(covariance_proj)) * 1e-6
                cholesky_factor = np.linalg.cholesky(covariance_proj)
                z = scipy.linalg.solve_triangular(
                    cholesky_factor, d.T, lower=True, check_finite=False,
                    overwrite_b=True)
                squared_maha = np.sum(z * z, axis=0)
                return squared_maha
        else:
            raise ValueError('invalid distance metric')



