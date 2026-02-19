GCN Syntax
==========

Like the R package gEcon, gEconpy uses .GCN files to represent a DSGE model. A GCN file is divided into blocks, each of which represents an optimization problem. Here is one block from the example Real Business Cycle (RBC) model included in the package.


An Example Model
----------------
Here is an example of a GCN file that represents a simple RBC model, found in the ``examples/GCN files`` directory:

.. code-block:: bash

    block HOUSEHOLD
    {
        definitions
        {
            u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C) -
                  L[] ^ (1 + sigma_L) / (1 + sigma_L);
        };

        controls
        {
            C[], L[], I[], K[];
        };

        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };

        constraints
        {
            C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
            K[] = (1 - delta) * K[-1] + I[] : q[];
        };

        calibration
        {
            # Fixed parameters
            beta  = 0.99;
            delta = 0.02;

            # Parameters to estimate
            sigma_C ~ N(loc=1.5, scale=0.1, lower=1.0) = 1.5;
            sigma_L ~ N(loc=2.0, scale=0.1, lower=1.0) = 2.0;
        };
    };

    block FIRM
    {
        controls
        {
            K[-1], L[];
        };

        objective
        {
            TC[] = -(r[] * K[-1] + w[] * L[]);
        };

        constraints
        {
            Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];
        };

        identities
        {
            # Perfect competition
            mc[] = 1;
        };

        calibration
        {
            alpha = 0.35;
        };
    };

    block TECHNOLOGY_SHOCKS
    {
        identities
        {
            log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
        };

        shocks
        {
            epsilon_A[];
        };

        calibration
        {
            rho_A = 0.95;
        };
    };


The file is structured around the optimization problems of the agents in the model. In this case, there are two agents: households and firms. We also add a separate block for the exogenous technology process, to make it clear that it is not part of either agent. The model corresponds to the following equations:


.. math::

    \begin{align}
        \text{Household} & \\
        & \max_{C_t, L_t, I_t, K_t} U_t = \sum_{t=0}^\infty \beta^t \left ( \frac{C_t^{1 - \sigma_C}}{1 - \sigma_C} - \frac{L_t^{1 + \sigma_L}}{1 + \sigma_L} \right ) \\
        \text{Subject to} & \\
        & C_t + I_t = r_t K_{t-1} + w_t L_t \\
        & K_t = (1 - \delta) K_{t-1} + I_t \\
        && \\
        \text{Firm} & \\
        & \min_{K_{t-1}, L_t} TC_t = r_t K_{t-1} + w_t L_t \\
        \text{Subject to} & \\
        & Y_t = A_t K_{t-1}^\alpha L_t^{1 - \alpha} \\
        && \\
        \text{Technology Shock} & \\
        & A_t = \rho_A A_{t-1} + \epsilon_{A_t}
    \end{align}


Explanation of GCN Syntax
-------------------------

A .GCN file uses an easy-to-read syntax. Whitespace is not meaningful - lines are terminated with a ";", so long equations can be split into multiple lines for readability. There are no reserved keywords* to avoid -- feel free to write beta instead of betta!

Model variables are written with a name followed by square brackets, as in ``U[]``. The square brackets give the time index the variable enters with. Following Dynare conventions, capital stock ``K[-1]`` enters with a lag, while all other variables enter in the present. Expectations are denoted by wrapping variables with ``E[]``, as in ``E[U[1]]`` for expected utility at t+1. So I lied, there are a couple reserved keywords (hence the asterisk). Finally, you can refer directly to steady-state values as ``K[ss]``.

Parameters are written exactly as variables, except they have no square brackets ``[]``.

Anatomy of a Block
Blocks are divided into five components: ``definitions``, ``controls``, ``objective``, ``constraints``, ``identities``, ``shocks``, and, ``calibration``. In this block, we see five of the seven. The blocks have the following functions:


Definitions
***********

``definitions`` contains equations that are **not** stored in the model. Instead, they are immediately substituted into all equations **within the same block**. In this example, a definition is used for the instantaneous utility function. It will be immediately substituted into the Bellman equation written in the `objective` block. Importantly, variables defined in the ``definitions`` block will not be available in other blocks!

Here is the definiton block used in the RBC model above:

.. code-block:: bash

    definitions
    {
        u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C) -
              L[] ^ (1 + sigma_L) / (1 + sigma_L);
    };


This gives us a "helper" variable ``u[]`` that represents the single-period utility function. In this case, the agent's utility is described using a CRRA utility function.


Controls
********
``controls`` are the variables under the agent's control. The objective function represented by the block will be solved by forming a Lagrange function and taking derivatives with respect to the controls.

Here is the control block used in the RBC model above:

.. code-block:: bash

    controls
    {
        C[], L[], I[], K[];
    };

This block means that the agent is free to choose consumption, labor, investment, and the next-period capital stock. In mathematical notation, these are the variables that appear under the :math:`\max` operator in the agent's utility function.


Objective
*********

The ``objective`` block contains only a single equation, and gives the function an agent will try to maximize over an infinite time horizon. In this case, the agent has a CRRA utility function.

In the above example, there are two examples of objectives. The household seeks to maximize utility. This is represented by the following block:

.. code-block:: bash

    objective
    {
        U[] = u[] + beta * E[][U[1]];
    };

In the mathematical notation above, this was written with an infinite sum. These sums cannot be represented in the GCN file, so the user must first convert the sum to a recursive Bellman equation. Notice also that the variable ``u[]``, defined in the ``definitions`` block, is used here to shorten the expression.

The firm seeks to minimize total costs. This is represented by the following block:

.. code-block:: bash

    objective
    {
        TC[] = -(r[] * K[-1] + w[] * L[]);
    };

In gEconpy, all objectives must be written as *maximization* problems. Since the firm is minimizing costs, the objective function is written with a minus sign.


Constraints
***********

``constraints`` give the resource constraints that the agent's maximization must respect. All constraints are given their own Lagrange multipliers.

In the above example, the household's optimization is done subject to a budget constraint and a law of motion of capital. These are represented by the following block:

.. code-block:: bash

    constraints
    {
        C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
        K[] = (1 - delta) * K[-1] + I[] : q[];
    };

The firm's minimization is done subject to a production technology, in this case Cobb-Douglas:

.. code-block:: bash

    constraints
    {
        Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];
    };

Identities
**********

``identities`` are equations that are not part of an optimization problem, but that are a part of the model. Unlike equations defined in the `definitions` block, `identities` are saved in the model's system of equations. Unlike equations in the ``constraints`` block, no Lagrange multipliers are assigned to identities.

The most important fact to know about identities that they are directly inserted into the model.

In the above example, there are two identity blocks. One is in the firm's problem, and gives the perfect competition condition:

.. code-block:: bash

    identities
    {
        # Perfect competition
        mc[] = 1;
    };

This implies that price is euqal to marginal cost, and that prices



Shocks
******
``shocks`` are exogenous variables that are not under the agent's control. In this case, the exogenous technology process is given in the ``TECHNOLOGY_SHOCKS`` block.

``shocks`` are where the user defines exogenous shocks, as in ``varexo`` in Dynare.


Calibration
***********
The ``calibration`` block where free parameters, calibrated parameters, and parameter prior distributions are defined.


Parameters
----------

All parameters must be given values. In general, there are three ways to assign a value: directly, by calibration, or by prior distribution.


Direct Assignment
******************
In the household block above, all parameters are given a value directly. In the ``HOUSEHOLD`` block above, the parameters ``beta`` and ``delta`` are given values directly in the calibration block, like this:

.. code-block:: bash

    calibration
    {
        # Fixed parameters
        beta  = 0.99;
        delta = 0.02;
    };


Calibration Equations
*********************

As an alternative to setting a parameter value directly, the user can declare a parameter to be calibrated. To do this, give a steady-state relationship that the parameter should be calibrated to ensure is true. The following GCN code block for the firm's optimization problem shows how this is done:

.. code-block:: bash

    calibration
    {
        L[ss] / K[ss] = 0.36 -> alpha;
    };

The ``alpha`` parameter is set so that in the steady state, the ratio of labor to capital is 0.36. On the back end, gEconpy will use an optimizer to find a value of ``alpha`` that satsifies the user's condition. Note that calibrated parameters cannot have prior distributions!


Prior Distributions
********************

The user can also assign a prior distribution to a parameter. This is done by using the ``~`` operator. The following GCN code block for the household's optimization problem shows how this is done:

.. code-block:: bash

    calibration
    {
        sigma_C ~ N(loc=1.5, scale=0.1, lower=1.0) = 1.5;
        sigma_L ~ N(loc=2.0, scale=0.1, lower=1.0) = 2.0;
    };

In this case, the parameters ``sigma_C`` and ``sigma_L`` are given normal prior distributions with means of 1.5 and 2.0, respectively. The standard deviation of the prior distribution is 0.1, and the lower bound is 1.0. The user can also specify an upper bound by adding ``upper=`` to the prior distribution declaration.

After the distribution, the user can also assign a starting value of each parameter. This is done by adding an equals sign and the starting value after the prior distribution declaration. In the example above, the starting value of ``sigma_C`` is 1.5, and the starting value of ``sigma_L`` is 2.0. If the user does not specify a starting value, gEconpy will use the mean of the prior distribution as the starting value. This starting value is used for all non-estimation tasks, such as solving for the inital steady state or the initial perturbation solution.


Lagrange Multipliers and First Order Conditions
------------------------------------------------
To solve the model and find the first order conditions, gEconpy uses the Lagrange multiplier method. Take the household problem stated above:

.. math::

    \begin{align}
        & \max_{C_t, L_t, I_t, K_t} U_t = \sum_{t=0}^\infty \beta^t \left ( \frac{C_t^{1 - \sigma_C}}{1 - \sigma_C} - \frac{L_t^{1 + \sigma_L}}{1 + \sigma_L} \right ) \\
        \text{Subject to} & \\
        & C_t + I_t = r_t K_{t-1} + w_t L_t \\
        & K_t = (1 - \delta) K_{t-1} + I_t \\
    \end{align}

This problem can be solved by forming the Lagrange function:

.. math::

    \mathcal{L} = \sum_{t=0}^\infty \beta^t \left ( \frac{C_t^{1 - \sigma_C}}{1 - \sigma_C} - \frac{L_t^{1 + \sigma_L}}{1 + \sigma_L} \right ) - \lambda_t (C_t + I_t - r_t K_{t-1} - w_t L_t) - q_t (K_t - (1 - \delta) K_{t-1} - I_t)

Then solving for the first order conditions with respect to the control variables:

.. math::

    \begin{align}
        & \frac{\partial \mathcal{L}}{\partial C_t} = 0 \Rightarrow C_t^{-\sigma_C} - \lambda_t = 0 \\
        & \frac{\partial \mathcal{L}}{\partial L_t} = 0 \Rightarrow -L_t^{\sigma_L} + \lambda_t w_t = 0 \\
        & \frac{\partial \mathcal{L}}{\partial I_t} = 0 \Rightarrow -\lambda_t + q_t = 0 \\
        & \frac{\partial \mathcal{L}}{\partial K_t} = 0 \Rightarrow -q_t + \beta \mathbb{E} \left [ \lambda_{t+1} r_{t+1} + q_{t+1} (1 - \delta) \right ] = 0 \\
    \end{align}


As mentioned, all constraints will automatically have a Lagrange multiplier assigned to them. The user name these multipliers himself by putting a colon ":" after an equation, followed by the Lagrange multipler name. Looking at the household constraints block fom above:

.. code-block:: bash

    constraints
    {
        C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
        K[] = (1 - delta) * K[-1] + I[] : q[];
    };

The multiplier associated with the budget constraint has been given the name "lambda", as is usual in the literature, while the law of motion of capital has been given the name ``q[]``. If the user wanted, she could use these variables in further computations within the block, for example ``Q[] = q[] / lambda[]``, Tobin's Q, could be added in the ``identities`` block.

Internally, first order conditions are solved by first making all substitutions from ``definitions``, then forming the following Lagrangian function:
``L = objective.RHS - lm1 * (control_1.LHS - control_1.RHS) - lm2 * (control_2.LHS - control_2.RHS) ... - lm_k * (control_k.LHS - control_k.RHS)``

Next, the derivative of this Lagrangian is taken with respect to all control variables and all lagrange multipliers. Derivaties are are computed "though time" using ``TimeAwareSymbols``, an extension of a normal Sympy symbol. For a control variable x, the total derivative over time is built up as ``dL[]/dx[] + beta * dL[+1]/dx + beta * beta * dL[+2]/dx[] ...``. This unrolling terminates when ``dL[+n]/dx[] = 0``.

The result of this unrolling and taking derivatives process are the first order conditions (FoC). All model FoCs, along with objectives, constraints, and identities, are saved into the system of equations that represents the model.
