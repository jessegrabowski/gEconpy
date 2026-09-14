Estimating a DSGE Model
=======================

gEconpy estimates a model by handing its linearized solution to a Kalman filter inside a PyMC model. The
parameters carry the priors declared in the GCN file, PyMC supplies the sampler, and pymc-extras supplies the state
space machinery. The estimation notebooks in the :doc:`example gallery </examples/gallery>` do this end to end on the
RBC model and on real data. This page names the four steps and what each one decides.

Build the state space model
---------------------------

:func:`~gEconpy.model.build.statespace_from_gcn` parses a GCN file the same way
:func:`~gEconpy.model.build.model_from_gcn` does, then wraps the result in a
:class:`~gEconpy.model.statespace.DSGEStateSpace`, which is a pymc-extras ``PyMCStateSpace``. Nothing is compiled
yet. The model still needs to know which of its variables are observed.

Configure it
------------

:meth:`~gEconpy.model.statespace.DSGEStateSpace.configure` fixes the settings that shape the likelihood, and it has to
be called before the graph is built. The choices that matter most:

``observed_states``
    Which model variables the data measure. A state space model can observe at most as many variables as it has
    sources of stochastic variation, so a model with one shock and no measurement error observes one variable.

``measurement_error``
    Which observed variables carry measurement error. Each gets its own error standard deviation as a parameter.

``constant_params``
    Parameters held at their GCN starting values and left out of the estimation.

``solver``
    How the linear policy function is found at every likelihood evaluation: ``gensys``, ``cycle_reduction``, or
    ``scan_cycle_reduction``. Only ``scan_cycle_reduction`` compiles under the JAX backend.

``mode``
    The PyTensor backend the likelihood is compiled to. ``"JAX"`` is the fast choice for sampling with NUTS.

``temporal_aggregation`` and ``observation_equations``
    How observed series relate to model variables when the data are at a lower frequency than the model, or when an
    observed series is a function of several variables. The mixed frequency estimation notebook covers both.

``configure`` prints a table of the parameters the PyMC model has to supply and the names it expects for them.

Declare priors
--------------

Inside a ``pm.Model`` context, :meth:`~gEconpy.model.statespace.DSGEStateSpace.to_pymc` creates a PyMC random
variable for every model parameter from the prior declared in the GCN file. Priors for the shock standard deviations
and any measurement error standard deviations are not in the GCN file and are declared by hand, under the names the
configure table listed. Priors can also be declared entirely by hand, without calling ``to_pymc``, as long as every
name in the table is present.

Before touching data, :func:`~gEconpy.model.statespace.data_from_prior` draws parameter values from those priors,
simulates a data set from each, and returns the values that generated it. Fitting the model to a simulated set and
recovering its parameters is the check that the model is identified and the priors are reasonable.

Build the graph and sample
--------------------------

:meth:`~gEconpy.model.statespace.DSGEStateSpace.build_statespace_graph`, called inside the same ``pm.Model``
context with a ``DataFrame`` of observed data, adds the Kalman filter likelihood to the model. Its keyword
arguments add penalty terms that push the sampler away from parameter values with no steady state or no unique
stable solution, and they are worth leaving on. From there the model is an ordinary PyMC model:
:func:`pymc.sample`, a Laplace approximation, or normalizing-flow adapted NUTS all work, and the estimation example
notebook compares them.

After sampling, ``sample_conditional_posterior`` and its prior counterpart, inherited from pymc-extras, recover
the filtered and smoothed states, and
:meth:`~gEconpy.model.statespace.DSGEStateSpace.sample_autocorrelation_matrices` and the other post-estimation
methods compute model moments under the posterior.

A compact version of the whole sequence, from the test suite:

.. literalinclude:: ../../../tests/model/test_statespace.py
   :language: python
   :pyobject: test_constant_params_excluded_from_prior_samples
