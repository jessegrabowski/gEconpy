Package Overview
================

Representing a DSGE Model
-------------------------
Like the R package gEcon, gEconpy uses .GCN files to represent a DSGE model. A GCN file is divided into blocks, each of which represents an optimization problem. Here is one block from the example Real Business Cycle (RBC) model included in the package.

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


Basic Basics
------------

A .GCN file uses an easy-to-read syntax. Whitespace is not meaningful - lines are terminated with a ";", so long equations can be split into multiple lines for readability. There are no reserved keywords* to avoid -- feel free to write beta instead of betta!

Model variables are written with a name followed by square brackets, as in ``U[]``. The square brackets give the time index the variable enters with. Following Dynare conventions, capital stock ``K[-1]`` enters with a lag, while all other variables enter in the present. Expectations are denoted by wrapping variables with ``E[]``, as in ``E[U[1]]`` for expected utility at t+1. So I lied, there are a couple reserved keywords (hence the asterisk). Finally, you can refer directly to steady-state values as ``K[ss]``.

Parameters are written exactly as variables, except they have no square brackets ``[]``.

Anatomy of a Block
Blocks are divided into five components: ``definitions``, ``controls``, ``objective``, ``constraints``, ``identities``, ``shocks``, and, ``calibration``. In this block, we see five of the seven. The blocks have the following functions:

1. ``definitions`` contains equations that are **not** stored in the model. Instead, they are immediately substituted into all equations **within the same block**. In this example, a definition is used for the instantaneous utility function. It will be immediately substitutited into the Bellman equation written in the `objective` block.
2. ``controls`` are the variables under the agent's control. The objective function represented by the block will be solved by forming a Lagrange function and taking derivatives with respect to the controls.
3. The ``objective`` block contains only a single equation, and gives the function an agent will try to maximize over an infinite time horizon. In this case, the agent has a CRRA utility function.
4. ``constraints`` give the resource constraints that the agent's maximization must respect. All constraints are given their own Lagrange multipiers.
5. ``identities`` are equations that are not part of an optimization problem, but that are a part of the model. Unlike equations defined in the `definitions` block, `identities` are saved in the model's system of equations.
6. ``shocks`` are where the user defines exogenous shocks, as in ``varexo`` in Dynare.
7. The ``calibration`` block where free parameters, calibrated parameters, and parameter prior distributions are defined.

## Parameter Values and Priors

All parameters must be given values. In the household block above, all parameters are given a value directly. ``beta`` and ``delta`` are set fixed, while ``sigma_C`` and ``sigma_L`` are given priors and starting values. The ``~`` operator denotes a Prior, while ``=`` denotes a fixed value. All parameters must have a fixed value -- this is used as the "default" value when building and solving the model. Priors, on the other hand, are optional. At present, the user can choose from ``Normal``, ``HalfNormal``, ``TruncatedNormal``, ``Beta``, ``Gamma``, ``Inverse_Gamma``, and ``Uniform`` priors, with more to come as I improve the integration with PyMC. Distributons can be parameterized either using the ``loc``, ``scale``, ``shape`` synatx of ``scipy.stats``, or directly using the common parameter values from the literature (such as a ``mu`` and ``sigma`` for a normal).


As an alternative to setting a parameter value directly, the user can declare a parameter to be calibrated. To do this, give a steady-state relationship that the parameter should be calibrated to ensure is true. The following GCN code block for the firm's optimization problem shows how this is done:

.. code-block:: bash

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
        L[ss] / K[ss] = 0.36 -> alpha;
        };
    };


The ``alpha`` parameter is set so that in the steady state, the ratio of labor to capital is 0.36. On the back end, gEconpy will use an optimizer to find a value of ``alpha`` that satsifies the user's condition. Note that calibrated parameters cannot have prior distributions!

Lagrange Multipliers and First Order Conditions
------------------------------------------------

As mentioned, all constraints will automatically have a Lagrange multiplier assigned to them. The user name these multipliers himself by putting a colon ":" after an equation, followed by the Lagrange multipler name. From the code above:

.. code-block:: bash

    C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
    K[] = (1 - delta) * K[-1] + I[] : q[];


The multiplier associated with the budget constraint has been given the name "lambda", as is usual in the literature, while the law of motion of capital has been given the name ``q[]``. If the user wanted, she could use these variables in further computations within the block, for example ``Q[] = q[] / lambda[]``, Tobin's Q, could be added in the ``identities`` block.

Interally, first order conditions are solved by first making all substitutions from ``definitions``, then forming the following Lagrangian function:
``L = objective.RHS - lm1 * (control_1.LHS - control_1.RHS) - lm2 * (control_2.LHS - control_2.RHS) ... - lm_k * (control_k.LHS - control_k.RHS)``

Next, the derivative of this Lagrangian is taken with respect to all control variables and all lagrange multipliers. Derivaties are are computed "though time" using ``TimeAwareSymbols``, an extension of a normal Sympy symbol. For a control variable x, the total derivative over time is built up as ``dL[]/dx[] + beta * dL[+1]/dx + beta * beta * dL[+2]/dx[] ...``. This unrolling terminates when ``dL[+n]/dx[] = 0``.

The result of this unrolling and taking derivatives process are the first order conditions (FoC). All model FoCs, along with objectives, constraints, and identities, are saved into the system of equations that represents the model.
