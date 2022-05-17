# gEcon.py
A collection of tools for working with DSGE models in python, inspired by the fantastic R package gEcon, http://gecon.r-forge.r-project.org/.

Like gEcon, gEcon.py solves first order conditions automatically, helping the researcher avoid math errors while facilitating rapid prototyping of models. By working in the optimization problem space rather than the FoC space, modifications to the model are much simpler. Adding an additional term to the utility function, for example, requires modifying only 2-3 lines of code, whereas in FoC space it may require re-solving the entire model by hand.

gEcon.py uses the GCN file originally created for the gEcon package. gEcon GCN files are fully compatable with gEcon.py, and includes all the great features of GCN files, including:
* Automatically solve first order conditions
* Users can include steady-state values in equations without explictly solving for them by hand first!
* Users can declare "calibrated parameters", requesting a parameter value be found to induce a specific steady-state relationship

gEcon.py is still in an unfinished alpha state, but I encourage anyone interested in DSGE modeling to give it a try and and report any bugs you might find.

## To Do List:
1. Bayesian and ML Estimation
2. Local identification tests
3. Improve symbolic simplification routines
4. Re-write entire back-end using Aesara, integrate with PyMC and allow NUTS sampling
5. Higher order linear approximations
6. Re-write and robustify steady-state solver
7. More robust GCN parser with clearer error messages
8. More diagnostic tools for debugging GCN file code

If you want to help with the project, please don't hesitate to reach out!

# Representing a DSGE Model
Like the R package gEcon, gEcon.py uses .GCN files to represent a DSGE model. A GCN file is divided into blocks, each of which represents an optimization problem. Here is one block from the example Real Business Cycle (RBC) model included in the package.

```
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
```
## Basic Basics
A .GCN file uses an easy-to-read syntax. Whitespace is not meaningful - lines are terminated with a ";", so long equations can be split into multiple lines for readability. There are no reserved keywords* to avoid -- feel free to write beta instead of betta!

Model variables are written with a name followed by square brackets, as in `U[]`. The square brackets give the time index the variable enters with. Following Dynare conventions, capital stock `K[-1]` enters with a lag, while all other variables enter in the present. Expectations are denoted by wrapping variables with `E[]`, as in `E[U[1]]` for expected utility at t+1. So I lied, there are a couple reserved keywords (hence the asterisk). Finally, you can refer directly to steady-state values as `K[ss]`. 

Parameters are written exactly as variables, except they have no square brackets `[]`. 

## Anatomy of a Block
Blocks are divided into five components: `definitions`, `controls`, `objective`, `constraints`, `identities`, `shocks`, and, `calibration`. In this block, we see five of the seven. The blocks have the following functions:

1. `definitions` contains equations that are **not** stored in the model. Instead, they are immediately substituted into all equations **within the same block**. In this example, a definition is used for the instantaneous utility function. It will be immediately substitutited into the Bellman equation written in the `objective` block.
2. `controls` are the variables under the agent's control. The objective function represented by the block will be solved by forming a Lagrange function and taking derivatives with respect to the controls.
3. The `objective` block contains only a single equation, and gives the function an agent will try to maximize over an infinite time horizon. In this case, the agent has a CRRA utility function.
4. `constraints` give the resource constraints that the agent's maximization must respect. All constraints are given their own Lagrange multipiers.
5. `identities` are equations that are not part of an optimization problem, but that are a part of the model. Unlike equations defined in the `definitions` block, `identities` are saved in the model's system of equations.
6. `shocks` are where the user defines exogenous shocks, as in `varexo` in Dynare.
7. The `calibration` block where free parameters, calibrated parameters, and parameter prior distributions are defined. 

## Parameter Values and Priors

All parameters must be given values. In the household block above, all parameters are given a value directly. `beta` and `delta` are set fixed, while `sigma_C` and `sigma_L` are given priors and starting values. The `~` operator denotes a Prior, while `=` denotes a fixed value. All parameters must have a fixed value -- this is used as the "default" value when building and solving the model. Priors, on the other hand, are optional. At present, the user can choose from `Normal`, `HalfNormal`, `TruncatedNormal`, `Beta`, `Gamma`, `Inverse_Gamma`, and `Uniform` priors, with more to come as I improve the integration with PyMC. Distributons can be parameterized either using the `loc`, `scale`, `shape` synatx of `scipy.stats`, or directly using the common parameter values from the literature (such as a `mu` and `sigma` for a normal).


As an alterantive to setting a parameter value directly, the user can declare a parameter to be calibrated. To do this, give a steady-state relationship that the parameter should be calibrated to ensure is true. The following GCN code block for the firm's optimization problem shows how this is done:

```
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
```
The `alpha` parameter is set so that in the steady state, the ratio of labor to capital is 0.36. On the back end, gEcon.py will use an optimizer to find a value of `alpha` that satsifies the user's condition. Note that calibrated parameters cannot have prior distributions! 

## Lagrange Multipliers and First Order Conditions
As mentioned, all constraints will automatically have a Lagrange multiplier assigned to them. The user name these multipliers himself by putting a colon ":" after an equation, followed by the Lagrange multipler name. From the code above:

```
C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
K[] = (1 - delta) * K[-1] + I[] : q[];
```

The multiplier associated with the budget constraint has been given the name "lambda", as is usual in the literature, while the law of motion of capital has been given the name `q[]`. If the user wanted, she could use these variables in further computations within the block, for example `Q[] = q[] / lambda[]`, Tobin's Q, could be added in the `identities` block.

Interally, first order conditions are solved by first making all substitutions from `definitions`, then forming the following Lagrangian function:
`L = objective.RHS - lm1 * (control_1.LHS - control_1.RHS) - lm2 * (control_2.LHS - control_2.RHS) ... - lm_k * (control_k.LHS - control_k.RHS)`

Next, the derivative of this Lagrangian is taken with respect to all control variables and all lagrange multipliers. Derivaties are are computed "though time" using `TimeAwareSymbols`, an extension of a normal Sympy symbol. For a control variable x, the total derivative over time is built up as `dL[]/dx[] + beta * dL[+1]/dx + beta * beta * dL[+2]/dx[] ...`. This unrolling terminates when `dL[+n]/dx[] = 0`. 

The result of this unrolling and taking derivatives process are the first order conditions (FoC). All model FoCs, along with objectives, constraints, and identities, are saved into the system of equations that represents the model.

## Steady State

After finding FoCs, the system will be ready to find a steady state and solve for a first-order linear approximation. To help the process, the user can write a `STEADY_STATE` block. This is a special reserved keyword block that can be placed anywhere in the GCN file. It should contain only `defintions` and `identities` as components. Here is an example of a steady state block for the RBC model:

```
block STEADY_STATE
{
    definitions
    {
      # If this is empty you can delete this, but it is also nice for writing common parameter or variable combinations.
    };

    identities
    {
        A[ss] = 1;
        P[ss] = 1;
        r[ss] = P[ss] * (1 / beta - (1 - delta));
        w[ss] = (1 - alpha) * P[ss] ^ (1 / (1 - alpha)) * (alpha / r[ss]) ^ (alpha / (1 - alpha));
        Y[ss] = (r[ss] / (r[ss] - delta * alpha)) ^ (sigma_C / (sigma_C + sigma_L)) *
            (w[ss] / P[ss] * (w[ss] / P[ss] / (1 - alpha)) ^ sigma_L) ^ (1 / (sigma_C + sigma_L));

        I[ss] = (delta * alpha / r[ss]) * Y[ss];
        C[ss] = Y[ss] ^ (-sigma_L / sigma_C) * ((1 - alpha) ^ (-sigma_L) * (w[ss] / P[ss]) ^ (1 + sigma_L)) ^ (1 / sigma_C);
        K[ss] = alpha * Y[ss] * P[ss] / r[ss];
        L[ss] = (1 - alpha) * Y[ss] * P[ss] / w[ss];


        U[ss] = (1 / (1 - beta)) * (C[ss] ^ (1 - sigma_C) / (1 - sigma_C) - L[ss] ^ (1 + sigma_L) / (1 + sigma_L));
        lambda[ss] = C[ss] ^ (-sigma_C) / P[ss];
        q[ss] = lambda[ss];
        TC[ss] = -(r[ss] * K[ss] + w[ss] * L[ss]);
    };
};
```

It is not necessary to write an empty `definitions` component; this was done just to show where it goes. All information from the model block, including parameters and variables, are available to use in the `STEADY_STATE` block regardless of where they appear relative to each other (you can put the STEADY_STATE block at the top if you wish -- the file is not parsed top-to-bottom). 

Note that these equations are not checked in any way -- if you put something in the `STEADY_STATE` block, it is taken as the Word of God, and model solving proceeds from there. If you are having trouble finding a steady state, be sure to double check these equations. 

Finally, you **do not** have to provide the complete steady state system! You can include only equations, and the rest will be passed to an optimizer to be solved. 

## Solving the model

Once a GCN file is written, using gEcon to do analysis is easy, as this code block shows:
```python
file_path = 'GCN Files/RBC_basic.gcn'
model = gEconModel(file_path, verbose=True)
```

When the model is loaded, you will get a message about the number of equations and variables, as well as some other basic model descriptions. You can then solve for the stead state:
```python
model.steady_state()
>>> Steady state found! Sum of squared residuals is 2.9196536232567403e-19
```

And get the linearized state space representation

```python
model.solve_model()
>>>Solution found, sum of squared residuals:  7.075155451456433e-30
>>>Norm of deterministic part: 0.000000000
>>>Norm of stochastic part:    0.000000000
```

To see how to do simulations, IRFs, and compute moments, see the example notebook.

# Other Features

## Dynare Code Generation

Since Dynare is still the gold standard in DSGE modeling, and this is a wacky open source package written by a literally who?, gEcon.py has the ability to automatically convert a solved model into a Dynare mod file. This is done as follows:

```python
from gEcon.shared.dynare_convert import make_mod_file
print(make_mod_file(model))
```

Output:
```
var A, C, I, K, L, TC, U, Y, mc, q, r, var_lambda, w;
varexo epsilon_A;

parameters param_alpha, param_beta, param_delta, rho_A;
parameters sigma_C, sigma_L;

param_alpha = 0.35;
param_beta = 0.99;
param_delta = 0.02;
rho_A = 0.95;
sigma_C = 1.5;
sigma_L = 2.0;

model;
-C - I + K(-1) * r + L * w = 0;
I - K + K(-1) *(1 - param_delta) = 0;
C ^(1 - sigma_C) /(1 - sigma_C) - L ^(sigma_L + 1) /(sigma_L + 1) - U + U(1) * param_beta = 0;
-var_lambda + C ^(- sigma_C) = 0;
-L ^ sigma_L + var_lambda * w = 0;
q - var_lambda = 0;
param_beta *(q(1) *(1 - param_delta) + r(1) * var_lambda(1)) - q = 0;
1 - mc = 0;
A * K(-1) ^ param_alpha * L ^(1 - param_alpha) - Y = 0;
-K(-1) * r - L * w - TC = 0;
A * K(-1) ^(param_alpha - 1) * L ^(1 - param_alpha) * param_alpha * mc - r = 0;
A * K(-1) ^ param_alpha * mc *(1 - param_alpha) / L ^ param_alpha - w = 0;
epsilon_A + rho_A * log(A(-1)) - log(A) = 0;
end;

initval;
A = 1.0000;
C = 2.3584;
I = 0.7146;
K = 35.7323;
L = 0.8201;
TC = -3.0731;
U = -148.6156;
Y = 3.0731;
var_lambda = 0.2761;
mc = 1.0000;
q = 0.2761;
r = 0.0301;
w = 2.4358;
end;

steady;
check(qz_zero_threshold=1e-20);

shocks;
var epsilon_A;
stderr 0.01;
end;

stoch_simul(order=1, irf=100, qz_zero_threshold=1e-20);
```

### Warings about Dynare Code
* No efforts are made to provide symbolic solutions to the steady-state! If you include steady state values in your equations and do not provide a symbolic solution in the STEADY_STATE block, the .mod file will not work "out of the box".
* If your model includes calibrated equations, the generated Dynare code **will not** work out of the box. You need to analyically compute the steady state values and add a deterministic relationship (that beings with `#`) to the model block.


## Estimation

Coming soon!
