from common import *

from collections import OrderedDict
class SimpleRBC(sm.tsa.statespace.MLEModel):

    parameters = OrderedDict([
        ('discount_rate', 0.95),
        ('disutility_labor', 3.),
        ('depreciation_rate', 0.025),
        ('capital_share', 0.36),
        ('technology_shock_persistence', 0.85),
        ('technology_shock_var', 0.04**2)
    ])

    def __init__(self, endog, calibrated=None):
        super(SimpleRBC, self).__init__(
            endog, k_states=2, k_posdef=1, initialization='stationary')
        self.k_predetermined = 1

        # Save the calibrated vs. estimated parameters
        parameters = self.parameters.keys()
        calibrated = calibrated or {}
        self.calibrated = OrderedDict([
            (param, calibrated[param]) for param in parameters
            if param in calibrated
        ])
        self.idx_calibrated = np.array([
            param in self.calibrated for param in parameters])
        self.idx_estimated = ~self.idx_calibrated

        self.k_params = len(self.parameters)
        self.k_calibrated = len(self.calibrated)
        self.k_estimated = self.k_params - self.k_calibrated

        self.idx_cap_share = parameters.index('capital_share')
        self.idx_tech_pers = parameters.index('technology_shock_persistence')
        self.idx_tech_var = parameters.index('technology_shock_var')

        # Setup fixed elements of system matrices
        self['selection', 1, 0] = 1

    @property
    def start_params(self):
        structural_params = np.array(self.parameters.values())[self.idx_estimated]
        measurement_variances = [0.1] * 3
        return np.r_[structural_params, measurement_variances]

    @property
    def param_names(self):
        structural_params = np.array(self.parameters.keys())[self.idx_estimated]
        measurement_variances = ['%s.var' % name for name in self.endog_names]
        return structural_params.tolist() + measurement_variances

    def log_linearize(self, params):
        # Extract the parameters
        (discount_rate, disutility_labor, depreciation_rate, capital_share,
         technology_shock_persistence, technology_shock_var) = params

        # Temporary values
        tmp = (1. / discount_rate - (1. - depreciation_rate))
        theta = (capital_share / tmp)**(1. / (1. - capital_share))
        gamma = 1. - depreciation_rate * theta**(1. - capital_share)
        zeta = capital_share * discount_rate * theta**(capital_share - 1)

        # Coefficient matrices from linearization
        A = np.eye(2)

        B11 = 1 + depreciation_rate * (gamma / (1 - gamma))
        B12 = (-depreciation_rate *
               (1 - capital_share + gamma * capital_share) /
               (capital_share * (1 - gamma)))
        B21 = 0
        B22 = capital_share / (zeta + capital_share*(1 - zeta))
        B = np.array([[B11, B12], [B21, B22]])

        C1 = depreciation_rate / (capital_share * (1 - gamma))
        C2 = (zeta * technology_shock_persistence /
              (zeta + capital_share*(1 - zeta)))
        C = np.array([[C1], [C2]])

        return A, B, C

    def solve(self, params):
        capital_share = params[self.idx_cap_share]
        technology_shock_persistence = params[self.idx_tech_pers]

        # Get the coefficient matrices from linearization
        A, B, C = self.log_linearize(params)

        # Jordan decomposition of B
        eigvals, right_eigvecs = np.linalg.eig(np.transpose(B))
        left_eigvecs = np.transpose(right_eigvecs)

        # Re-order, ascending
        idx = np.argsort(eigvals)
        eigvals = np.diag(eigvals[idx])
        left_eigvecs = left_eigvecs[idx, :]

        # Blanchard-Khan conditions
        k_nonpredetermined = self.k_states - self.k_predetermined
        k_stable = len(np.where(eigvals.diagonal() < 1)[0])
        k_unstable = self.k_states - k_stable
        if not k_stable == self.k_predetermined:
            raise RuntimeError('Blanchard-Kahn condition not met.'
                               ' Unique solution does not exist.')

        # Create partition indices
        k = self.k_predetermined
        p1 = np.s_[:k]
        p2 = np.s_[k:]

        p11 = np.s_[:k, :k]
        p12 = np.s_[:k, k:]
        p21 = np.s_[k:, :k]
        p22 = np.s_[k:, k:]

        # Decouple the system
        decoupled_C = np.dot(left_eigvecs, C)

        # Solve the explosive component (controls) in terms of the
        # non-explosive component (states) and shocks
        tmp = np.linalg.inv(left_eigvecs[p22])

        # This is \phi_{ck}, above
        policy_state = - np.dot(tmp, left_eigvecs[p21]).squeeze()
        # This is \phi_{cz}, above
        policy_shock = -(
            np.dot(tmp, 1. / eigvals[p22]).dot(
                np.linalg.inv(
                    np.eye(k_nonpredetermined) -
                    technology_shock_persistence / eigvals[p22]
                )
            ).dot(decoupled_C[p2])
        ).squeeze()

        # Solve for the non-explosive transition
        # This is T_{kk}, above
        transition_state = np.squeeze(B[p11] + np.dot(B[p12], policy_state))
        # This is T_{kz}, above
        transition_shock = np.squeeze(np.dot(B[p12], policy_shock) + C[p1])

        # Create the full design matrix
        tmp = (1 - capital_share) / capital_share
        tmp1 = 1. / capital_share
        design = np.array([[1 - tmp * policy_state, tmp1 - tmp * policy_shock],
                           [1 - tmp1 * policy_state, tmp1 * (1-policy_shock)],
                           [policy_state,            policy_shock]])

        # Create the transition matrix
        transition = (
            np.array([[transition_state, transition_shock],
                      [0,                technology_shock_persistence]]))

        return design, transition

    def transform_discount_rate(self, param, untransform=False):
        # Discount rate must be between 0 and 1
        epsilon = 1e-4  # bound it slightly away from exactly 0 or 1
        if not untransform:
            return np.abs(1 / (1 + np.exp(param)) - epsilon)
        else:
            return np.log((1 - param + epsilon) / (param + epsilon))

    def transform_disutility_labor(self, param, untransform=False):
        # Disutility of labor must be positive
        return param**2 if not untransform else param**0.5

    def transform_depreciation_rate(self, param, untransform=False):
        # Depreciation rate must be positive
        return param**2 if not untransform else param**0.5

    def transform_capital_share(self, param, untransform=False):
        # Capital share must be between 0 and 1
        epsilon = 1e-4  # bound it slightly away from exactly 0 or 1
        if not untransform:
            return np.abs(1 / (1 + np.exp(param)) - epsilon)
        else:
            return np.log((1 - param + epsilon) / (param + epsilon))

    def transform_technology_shock_persistence(self, param, untransform=False):
        # Persistence parameter must be between -1 and 1
        if not untransform:
            return param / (1 + np.abs(param))
        else:
            return param / (1 - param)

    def transform_technology_shock_var(self, unconstrained, untransform=False):
        # Variances must be positive
        return unconstrained**2 if not untransform else unconstrained**0.5

    def transform_params(self, unconstrained):
        constrained = np.zeros(unconstrained.shape, unconstrained.dtype)

        i = 0
        for param in self.parameters.keys():
            if param not in self.calibrated:
                method = getattr(self, 'transform_%s' % param)
                constrained[i] = method(unconstrained[i])
                i += 1

        # Measurement error variances must be positive
        constrained[self.k_estimated:] = unconstrained[self.k_estimated:]**2

        return constrained

    def untransform_params(self, constrained):
        unconstrained = np.zeros(constrained.shape, constrained.dtype)

        i = 0
        for param in self.parameters.keys():
            if param not in self.calibrated:
                method = getattr(self, 'transform_%s' % param)
                unconstrained[i] = method(constrained[i], untransform=True)
                i += 1

        # Measurement error variances must be positive
        unconstrained[self.k_estimated:] = constrained[self.k_estimated:]**0.5

        return unconstrained

    def update(self, params, **kwargs):
        params = super(SimpleRBC, self).update(params, **kwargs)

        # Reconstruct the full parameter vector from the
        # estimated and calibrated parameters
        structural_params = np.zeros(self.k_params, dtype=params.dtype)
        structural_params[self.idx_calibrated] = self.calibrated.values()
        structural_params[self.idx_estimated] = params[:self.k_estimated]
        measurement_variances = params[self.k_estimated:]

        # Solve the model
        design, transition = self.solve(structural_params)

        # Update the statespace representation
        self['design'] = design
        self['obs_cov', 0, 0] = measurement_variances[0]
        self['obs_cov', 1, 1] = measurement_variances[1]
        self['obs_cov', 2, 2] = measurement_variances[2]
        self['transition'] = transition
        self['state_cov', 0, 0] = structural_params[self.idx_tech_var]
