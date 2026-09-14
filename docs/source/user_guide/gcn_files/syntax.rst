GCN Syntax
==========

A GCN file describes a DSGE model as a set of blocks. Each block is one agent's optimization problem, or a group of
equations that belong together, such as an exogenous process. gEconpy derives the first-order conditions of every
optimization block, collects them with the identities, and builds the model's system of equations from the result.
The format follows the R package gEcon, and files written for gEcon parse without changes.

Every sample on this page is read from the RBC model shipped with the package, ``gEconpy/data/GCN Files/RBC.gcn``,
which you can load with :func:`~gEconpy.data.examples.get_example_gcn`. The full file is at the end of the page.


Variables, parameters and time
------------------------------

Whitespace is not meaningful. Statements end with a semicolon, so a long equation can span several lines. Comments
start with ``#``.

A variable is a name followed by square brackets giving its time index: ``C[]`` is consumption at time :math:`t`,
``K[-1]`` is capital chosen in the previous period, and ``U[1]`` is next period's utility. Following Dynare, a
predetermined stock such as capital enters with a lag. ``E[][U[1]]`` is the expectation of ``U[1]`` conditional on
information at :math:`t`. ``K[ss]`` refers to the steady-state value of ``K``.

A parameter is a name with no square brackets, such as ``beta`` or ``alpha``. There are no reserved parameter
names, so ``beta`` is fine where Dynare would need ``betta``.


The household block
-------------------

The household block of the RBC model uses every component an optimization block can have:

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
   :start-at: block HOUSEHOLD
   :end-before: block FIRM

It represents the problem

.. math::

    \begin{align}
        & \max_{C_t, L_t, I_t, K_t} U_t = \sum_{t=0}^\infty \beta^t \left ( \frac{C_t^{1 - \sigma_C}}{1 - \sigma_C} - \frac{L_t^{1 + \sigma_L}}{1 + \sigma_L} \right ) \\
        \text{subject to} & \\
        & C_t + I_t = r_t K_{t-1} + w_t L_t \\
        & K_t = (1 - \delta) K_{t-1} + I_t
    \end{align}

The seven components a block can hold are described in turn.

``definitions``
    Helper equations that are substituted into the other equations of the same block and then discarded. They
    never enter the model's system of equations, and a name defined here is not visible in other blocks. The RBC
    household defines the period utility function once so the objective stays short:

    .. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
       :language: text
       :start-at: definitions
       :end-at: };
       :dedent: 1

``controls``
    The variables the agent chooses. gEconpy forms a Lagrangian and differentiates it with respect to each control.
    These are the variables under the :math:`\max` operator above:

    .. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
       :language: text
       :start-at: controls
       :end-at: };
       :dedent: 1

``objective``
    A single equation giving the function the agent maximizes over an infinite horizon. The infinite sum in the
    mathematical statement cannot be written directly, so the objective is written as a Bellman equation:

    .. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
       :language: text
       :start-at: objective
       :end-at: };
       :dedent: 1

``constraints``
    The equations the optimization must respect. Every constraint gets a Lagrange multiplier. Writing ``: name[]``
    after a constraint names its multiplier, and a multiplier you name can be used elsewhere in the block, for
    example in an identity such as ``Q[] = q[] / lambda[]`` for Tobin's Q. A constraint with no name still gets a
    multiplier with a generated name.

    .. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
       :language: text
       :start-at: constraints
       :end-at: };
       :dedent: 1

``identities``
    Equations that are part of the model but not of any optimization problem. Unlike definitions they are kept in
    the system of equations, and unlike constraints they get no multiplier. The firm block below uses one for the
    perfect-competition condition ``mc[] = 1``, and the technology block uses one for the shock process.

``shocks``
    The exogenous innovations, as ``varexo`` in Dynare. Each shock is a variable that gEconpy treats as a
    zero-mean disturbance:

    .. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
       :language: text
       :start-at: shocks
       :end-at: };
       :dedent: 1

``calibration``
    Values, priors and calibrating equations for the block's parameters. See `Parameters`_ below.


The firm and the technology process
-----------------------------------

The firm minimizes total cost subject to a Cobb-Douglas production technology. The RBC model writes the minimization
by negating the objective, which is what gEcon requires:

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
   :start-at: block FIRM
   :end-before: block TECHNOLOGY_SHOCKS

The ``@minimize`` tag is the clearer alternative. gEconpy negates the objective internally before forming the
Lagrangian, so the first-order conditions are those of the minimization program, and the variable keeps a positive
steady state that can be log-linearized:

.. literalinclude:: ../../../../tests/_resources/test_gcns/rbc_2_block_minimize.gcn
   :language: text
   :start-at: objective
   :end-at: };
   :dedent: 1

An explicit ``@maximize`` tag is accepted and is the default when no tag is present.

The technology process is not part of any agent's problem, so it gets a block of its own with an identity for the
law of motion, a shock, and the persistence parameter:

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
   :start-at: block TECHNOLOGY_SHOCKS

This is :math:`\log A_t = \rho_A \log A_{t-1} + \epsilon_{A,t}`.


Parameters
----------

Every parameter that appears in an equation needs a value in some ``calibration`` block, or the file fails to parse
with :class:`~gEconpy.exceptions.OrphanParameterError`. There are three ways to give one.

**A fixed value.** ``alpha = 0.35;`` sets the parameter and nothing else.

**A prior and a starting value.** The ``~`` operator attaches a prior distribution, and ``= value`` after it gives
the starting value gEconpy uses for every task that is not estimation, such as solving the steady state or the
perturbation. The starting value is required. A prior with no starting value leaves the parameter with no value,
and parsing fails. The RBC household calibration attaches a prior to each of its four parameters:

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
   :start-at: calibration
   :end-at: };
   :dedent: 1

Priors are `preliz <https://preliz.readthedocs.io/en/latest/>`_ distributions, written with preliz's names and
parameter names, for example ``Normal(mu=1.5, sigma=0.1)`` or ``Gamma(alpha=2, beta=1)``. Every distribution in
preliz's gallery is accepted. Four wrappers modify a distribution: ``maxent(dist, lower=, upper=, mass=)`` picks the
distribution's parameters so that ``mass`` of its probability lies between the bounds, which is how the RBC model
writes its priors; ``Truncated(dist, lower=, upper=)`` and ``Censored(dist, lower=, upper=)`` bound a
distribution's support; and ``Hurdle(dist, psi=)`` mixes it with a point mass at zero.

**A calibrating equation.** Instead of a value, give a steady-state relationship the parameter must satisfy, followed
by ``->`` and the parameter name. gEconpy solves for the parameter together with the steady state:

.. code-block:: text

    calibration
    {
        L[ss] / K[ss] = 0.36 -> alpha;
    };

Here ``alpha`` takes whatever value makes the steady-state labor to capital ratio equal 0.36. A calibrated parameter
cannot also have a prior.


Excluding an equation
---------------------

A constraint is sometimes needed to solve an agent's problem but redundant in the final system. The household budget
constraint is the usual case: with an aggregate resource constraint ``Y[] = C[] + I[]`` elsewhere in the model, Walras'
law makes the two redundant. The ``@exclude`` tag above a constraint keeps it for the first-order conditions and drops
it from the system. Its multiplier is still created and remains a model variable.

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC_two_household.gcn
   :language: text
   :start-at: @exclude
   :end-at: q[];
   :dedent: 2


The steady state block
----------------------

A block named ``STEADY_STATE`` (or ``SS``, ``STEADYSTATE``, ``STEADY``) holds analytical steady-state relationships.
It can go anywhere in the file, because the file is not read top to bottom, and it may contain ``definitions`` and
``identities`` only. Any variable or parameter of the model can be used in it. The RBC model gives its complete
steady state:

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
   :start-at: block STEADY_STATE
   :end-before: block HOUSEHOLD

The equations are taken as given and are not checked. If a steady state cannot be found, they are the first thing to
verify. A partial steady state is fine: the equations given are used, and the remaining variables go to the numerical
solver.


Special blocks
--------------

Three blocks apply to the whole file and sit outside any agent's block.

``tryreduce`` lists variables that gEconpy should try to eliminate from the system by substitution, which makes the
model smaller before it is solved. The RBC model removes the two objective values, which nothing else depends on:

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
   :start-at: tryreduce
   :end-at: };

``assumptions`` declares sympy assumptions for named variables and parameters, grouped by assumption. The assumptions
gEconpy accepts are ``positive``, ``negative``, ``nonpositive``, ``nonnegative``, ``real``, ``integer``, ``finite``,
``rational`` and ``irrational``. A ``positive`` declaration is what lets gEconpy simplify a power of the variable or
log-linearize it without a sign check:

.. literalinclude:: ../../../../tests/_resources/test_gcns/open_rbc.gcn
   :language: text
   :start-at: assumptions
   :end-before: block STEADY_STATE

``options`` holds file-level settings. ``linear = True;`` declares that the model is already linear, so gEconpy skips
linearization. The gEcon output options, such as ``output logfile = TRUE;``, are accepted so that gEcon files parse and
are otherwise ignored.


How first-order conditions are derived
--------------------------------------

For the household problem above, gEconpy forms the Lagrangian

.. math::

    \mathcal{L} = \sum_{t=0}^\infty \beta^t \left ( \frac{C_t^{1 - \sigma_C}}{1 - \sigma_C} - \frac{L_t^{1 + \sigma_L}}{1 + \sigma_L} \right ) - \lambda_t (C_t + I_t - r_t K_{t-1} - w_t L_t) - q_t (K_t - (1 - \delta) K_{t-1} - I_t)

and differentiates it with respect to each control:

.. math::

    \begin{align}
        & \frac{\partial \mathcal{L}}{\partial C_t} = 0 \Rightarrow C_t^{-\sigma_C} - \lambda_t = 0 \\
        & \frac{\partial \mathcal{L}}{\partial L_t} = 0 \Rightarrow -L_t^{\sigma_L} + \lambda_t w_t = 0 \\
        & \frac{\partial \mathcal{L}}{\partial I_t} = 0 \Rightarrow -\lambda_t + q_t = 0 \\
        & \frac{\partial \mathcal{L}}{\partial K_t} = 0 \Rightarrow -q_t + \beta \mathbb{E} \left [ \lambda_{t+1} r_{t+1} + q_{t+1} (1 - \delta) \right ] = 0
    \end{align}

Internally, the substitutions from ``definitions`` are made first, then the Lagrangian is built as the objective's
right-hand side minus each multiplier times its constraint written as ``lhs - rhs``. An objective tagged
``@minimize`` is negated first. The derivative with respect to a control is taken through time using
:class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol`: for a control :math:`x_t` the total derivative is
:math:`\partial \mathcal{L}_t / \partial x_t + \beta \, \partial \mathcal{L}_{t+1} / \partial x_t + \beta^2 \,
\partial \mathcal{L}_{t+2} / \partial x_t + \ldots`, and the sum stops at the first term that is identically zero.
The first-order conditions, the objectives, the constraints not tagged ``@exclude``, and the identities together form
the model's system of equations.


The complete RBC file
---------------------

.. literalinclude:: ../../../../gEconpy/data/GCN Files/RBC.gcn
   :language: text
