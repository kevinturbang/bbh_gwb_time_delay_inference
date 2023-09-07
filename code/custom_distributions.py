from numpyro.distributions.distribution import Distribution
from numpyro.distributions.continuous import Normal,HalfNormal
import jax
import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    validate_sample,
)

def build_ar(total,new_element):
    phi,w = new_element
    total = phi*total+w
    return total,total

class TransformedUniform(Distribution):

    arg_constraints = {"low": constraints.real, "high": constraints.real}
    support = constraints.real
    reparameterized_params = ["low","high"]

    def __init__(self, low=0.0, high=1.0, *, validate_args=None):

        self.low, self.high = promote_shapes(low, high)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        # Sample on the reals from a normal distribution
        logit_sample_unscaled = jax.random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
            )
        logit_sample = 2.5*logit_sample_unscaled

        # Transform to variable bounded on (low,high)
        exp_logit = jnp.exp(logit_sample)
        x = (exp_logit*self.high + self.low)/(1.+exp_logit)

        return x
        
    @validate_sample
    def log_prob(self,value):

        # Jacobian from unbounded logit variable to bounded sample
        dlogit_dx = 1./(value-self.low) + 1./(self.high-value)

        # Subtract Gaussian log-prob and apply Jacobian
        logit_value = jnp.log((value-self.low)/(self.high-value))
        return logit_value**2/(2.*2.5**2)-jnp.log(dlogit_dx)

class ARInitial(Distribution):

    arg_constraints = {"std_min": constraints.real, "std_high": constraints.real}
    support = constraints.real

    def __init__(self,deltas,std_scale=1.177,tau_min=0.5,tau_max=1.,validate_args=None):

        batch_shape = ()
        event_shape = jnp.shape(deltas)

        self._support = constraints.interval(tau_min, tau_max)
        self.deltas = deltas
        self.std_scale = std_scale
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Piecemeal distributions we'll sample from below
        self._normal = Normal(0.,1.)
        self._halfnormal = HalfNormal(1.)
        self._transformeduniform = TransformedUniform(self.tau_min,self.tau_max)

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self,key,sample_shape=()):
        assert is_prng_key(key)

        std_key,tau_key,ln_f_ref_key,steps_key = jax.random.split(key,4)

        # Draw variance parameter from a standard normal
        # Note that we'll assert an exp(-x**4) probability in `log_prob` below
        ar_std = self._halfnormal.sample(std_key,sample_shape=sample_shape + self.batch_shape)

        # Draw scale length parameter
        ar_tau = self._transformeduniform.sample(tau_key,sample_shape=sample_shape + self.batch_shape)

        # Draw inital value to seed AR process 
        ln_f_ref_unscaled = self._normal.sample(ln_f_ref_key,sample_shape=sample_shape+self.batch_shape)
        ln_f_ref = ln_f_ref_unscaled*ar_std

        ln_f_steps_unscaled = self._normal.sample(steps_key,sample_shape=sample_shape+self.batch_shape+self.event_shape)
        phis = jnp.exp(-self.deltas/ar_tau)
        ws = jnp.sqrt(-jnp.expm1(-2.*self.deltas/ar_tau))*ar_std*ln_f_steps_unscaled
        final,ln_fs = jax.lax.scan(build_ar,ln_f_ref,jnp.transpose(jnp.array([phis,ws])))
        ln_fs = jnp.append(ln_f_ref,ln_fs)
        ln_f_steps_unscaled = jnp.append(ln_f_ref_unscaled,ln_f_steps_unscaled)

        return ar_std,ar_tau,ln_fs,ln_f_steps_unscaled

    @validate_sample
    def log_prob(self,values):

        # Extract params
        ar_std,ar_tau,ln_fs,ln_f_steps_unscaled = values

        # Overwrite squared exponential with quadratic exponential
        logp = -self._halfnormal.log_prob(ar_std) - (ar_std/self.std_scale)**4
        logp += self._transformeduniform.log_prob(ar_tau)
        logp += jnp.sum(self._normal.log_prob(ln_f_steps_unscaled))

        return logp

class AR(Distribution):

    arg_constraints = {"std_min": constraints.real, "std_high": constraints.real}
    support = constraints.real

    def __init__(self,deltas,reference_index=0,std_scale=1.177,tau_min=0.5,tau_max=1.,validate_args=None):

        batch_shape = ()
        event_shape = jnp.shape(deltas)

        self._support = constraints.interval(tau_min, tau_max)
        self.deltas = deltas
        self.reference_index=reference_index
        self.std_scale = std_scale
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Piecemeal distributions we'll sample from below
        self._normal = Normal(0.,1.)
        self._halfnormal = HalfNormal(1.)
        self._transformeduniform = TransformedUniform(self.tau_min,self.tau_max)

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self,key,sample_shape=()):
        assert is_prng_key(key)

        std_key,tau_key,ln_f_ref_key,steps_key = jax.random.split(key,4)

        # Draw variance parameter from a standard normal
        # Note that we'll assert an exp(-x**4) probability in `log_prob` below
        ar_std = self._halfnormal.sample(std_key,sample_shape=sample_shape + self.batch_shape)

        # Draw scale length parameter
        ar_tau = self._transformeduniform.sample(tau_key,sample_shape=sample_shape + self.batch_shape)

        # Draw inital value to seed AR process 
        ln_f_ref_unscaled = self._normal.sample(ln_f_ref_key,sample_shape=sample_shape+self.batch_shape)
        ln_f_ref = ln_f_ref_unscaled*ar_std

        # Split deltas into forward and backward steps
        deltas_high = self.deltas[self.reference_index:]
        deltas_low_reversed = self.deltas[:self.reference_index][::-1]

        # Sample unscaled steps, split into forward and backward steps
        ln_f_steps_unscaled = self._normal.sample(steps_key,sample_shape=sample_shape+self.batch_shape+self.event_shape)
        ln_f_steps_unscaled_forward = ln_f_steps_unscaled[self.reference_index:]
        ln_f_steps_unscaled_backward = ln_f_steps_unscaled[:self.reference_index]

        # Build forward process
        phis_forward = jnp.exp(-deltas_high/ar_tau)
        ws_forward = jnp.sqrt(-jnp.expm1(-2.*deltas_high/ar_tau))*ar_std*ln_f_steps_unscaled_forward
        final,ln_fs_forward = jax.lax.scan(build_ar,ln_f_ref,jnp.transpose(jnp.array([phis_forward,ws_forward])))
        ln_fs = jnp.append(ln_f_ref,ln_fs_forward)

        # Build backward process
        phis_backward = jnp.exp(-deltas_low_reversed/ar_tau)
        ws_backward = jnp.sqrt(-jnp.expm1(-2.*deltas_low_reversed/ar_tau))*ar_std*ln_f_steps_unscaled_backward
        final,ln_fs_backward = jax.lax.scan(build_ar,ln_f_ref,jnp.transpose(jnp.array([phis_backward,ws_backward])))

        # Combine
        ln_fs = jnp.append(ln_fs_backward[::-1],ln_fs)
        ln_f_steps_unscaled = jnp.append(ln_f_ref_unscaled,ln_f_steps_unscaled)

        return ar_std,ar_tau,ln_fs,ln_f_steps_unscaled

    @validate_sample
    def log_prob(self,values):

        # Extract params
        ar_std,ar_tau,ln_fs,ln_f_steps_unscaled = values

        # Overwrite squared exponential with quadratic exponential
        logp = -self._halfnormal.log_prob(ar_std) - (ar_std/self.std_scale)**4
        logp += self._transformeduniform.log_prob(ar_tau)
        logp += jnp.sum(self._normal.log_prob(ln_f_steps_unscaled))

        return logp
