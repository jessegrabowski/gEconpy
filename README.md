# gEconpy
A collection of tools for working with DSGE models in python, inspired by the fantastic R package gEcon, http://gecon.r-forge.r-project.org/.

Like gEcon, gEconpy solves first order conditions automatically, helping the researcher avoid math errors while facilitating rapid prototyping of models. By working in the optimization problem space rather than the FoC space, modifications to the model are much simpler. Adding an additional term to the utility function, for example, requires modifying only 2-3 lines of code, whereas in FoC space it may require re-solving the entire model by hand.

gEconpy uses the GCN file originally created for the gEcon package. gEcon GCN files are fully compatible with gEconpy, and includes all the great features of GCN files, including:
* Automatically solve first order conditions
* Users can include steady-state values in equations without explictly solving for them by hand first!
* Users can declare "calibrated parameters", requesting a parameter value be found to induce a specific steady-state relationship

New features, which are not backwards compatible to R, also exist:

* Specify parameters as functions of other parameters
* Declare a partial steady state to help the optimizer more quickly find a solution
* Or, declare the entire analytic steady state, and skip the optimizer!
* Add priors to parameters, including a wide range of distributions and functions on distributions

gEconpy is still in an unfinished alpha state, but I encourage anyone interested in DSGE modeling to give it a try and and report any bugs you might find.

## Contributing:
Contributions from anyone are welcome, regardless of previous experience. Please check the Issues tab for open issues, or to create a new issue.

# Representing a DSGE Model
Like the R package gEcon, gEconpy uses .GCN files to represent a DSGE model. A GCN file is divided into blocks, each of which represents an optimization problem. Here the household block from a Real Business Cycle (RBC) model:

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
		delta = 0.02;

		# Parameters to estimate
	    beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;

		sigma_C ~ Truncated(Normal(mu=1.5, sigma=0.1), lower=1.0) = 1.5;
		sigma_L ~ Truncated(Nornal(mu=2.0, sigma=0.1), lower=1.0) = 2.0;
	};
};
```
## Basic Basics
A .GCN file uses an easy-to-read syntax. Whitespace is not meaningful - lines are terminated with a ";", so long equations can be split into multiple lines for readability. There are no reserved keywords* to avoid -- feel free to write beta instead of betta!

Model variables are written with a name followed by square brackets, as in `U[]`. The square brackets give the time index the variable enters with. Following Dynare conventions, capital stock `K[-1]` enters with a lag, while all other variables enter in the present. Expectations are denoted by wrapping variables with `E[]`, as in `E[U[1]]` for expected utility at t+1. So I lied, there are a couple reserved keywords (hence the asterisk). Finally, you can refer directly to steady-state values as `K[ss]`.

Parameters are written exactly as variables, except they have no square brackets `[]`.

## Anatomy of a Block
Blocks are divided into five components: `definitions`, `controls`, `objective`, `constraints`, `identities`, `shocks`, and, `calibration`. In this block, we see five of the seven. The blocks have the following functions:

1. `definitions` contains equations that are **not** stored in the model. Instead, they are immediately substituted into all equations **within the same block**. In this example, a definition is used for the instantaneous utility function. It will be immediately substituted into the Bellman equation written in the `objective` block.
2. `controls` are the variables under the agent's control. The objective function represented by the block will be solved by forming a Lagrange function and taking derivatives with respect to the controls.
3. The `objective` block contains only a single equation, and gives the function an agent will try to maximize over an infinite time horizon. In this case, the agent has a CRRA utility function.
4. `constraints` give the resource constraints that the agent's maximization must respect. All constraints are given their own Lagrange multipiers.
5. `identities` are equations that are not part of an optimization problem, but that are a part of the model. Unlike equations defined in the `definitions` block, `identities` are saved in the model's system of equations.
6. `shocks` are where the user defines exogenous shocks, as in `varexo` in Dynare.
7. The `calibration` block where free parameters, calibrated parameters, and parameter prior distributions are defined.

## Parameter Values and Priors

All parameters must be given values. In the household block above, all parameters are given a value directly. `beta` and `delta` are set fixed, while `sigma_C` and `sigma_L` are given priors and starting values. The `~` operator denotes a Prior, while `=` denotes a fixed value. All parameters must have a fixed value -- this is used as the "default" value when building and solving the model. Priors, on the other hand, are optional.

To represent priors, gEconpy uses the [preliz](https://preliz.readthedocs.io/en/latest/) package. All priors should follow preliz names and parameterization. For a full list, see their [example gallery](https://preliz.readthedocs.io/en/latest/gallery_content.html). Notice that:

* Distributions are capitalized, as in `Normal`, `Beta`, `Gamma`. For those with multiple words, use camel-case (e.g. `InverseGamma`)
* Parameters are given as keyword arguments, as in `Normal(mu=0, sigma=1)`.
* You are allowed to use preliz transformations on distributions, including [`Truncated`](https://preliz.readthedocs.io/en/latest/distributions/gallery/truncated.html), [`Censored`](https://preliz.readthedocs.io/en/latest/distributions/gallery/censored.html), [`Hurdle`](https://preliz.readthedocs.io/en/latest/distributions/gallery/hurdle.html), and [`Mixture`](https://preliz.readthedocs.io/en/latest/distributions/gallery/mixture.html). For example, `Truncated(Normal(mu=0, sigma=1), lower=0)` is a truncated normal distribution with a lower bound of 0.
* You can also directly call `maxent` to parameterize a distribution by an HDI range and probability mass within that range. For example, `beta ~ maxent(Beta(), lower=0.95, upper=0.99, mass=0.99)` finds the beta distribution with 99% of its mass between 0.95 and 0.99, and which is otherwise maximally uninformative.

As an alternative to setting a parameter value directly, the user can declare a parameter to be calibrated. To do this, give a steady-state relationship that the parameter should be calibrated to ensure is true. The following GCN code block for the firm's optimization problem shows how this is done:

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
The `alpha` parameter is set so that in the steady state, the ratio of labor to capital is 0.36. On the back end, gEconpy will use an optimizer to find a value of `alpha` that satisfies the user's condition. Note that calibrated parameters cannot have prior distributions!

## Lagrange Multipliers and First Order Conditions
As mentioned, all constraints will automatically have a Lagrange multiplier assigned to them. The user name these multipliers himself by putting a colon ":" after an equation, followed by a variable to be used for that Lagrange multiplier. From the code above:

```
C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
K[] = (1 - delta) * K[-1] + I[] : q[];
```

The multiplier associated with the budget constraint has been given the variable `lambda[]`, as is usual in the literature. The law of motion of capital has been given the variable `q[]`. If the user wanted, she could use these variables in further computations within the block, for example `Q[] = q[] / lambda[]`, Tobin's Q, could be added in the `identities` block.

Internally, first order conditions are solved by first making all substitutions from `definitions`, then forming the following Lagrangian function:
`L = objective.RHS - lm1 * (control_1.LHS - control_1.RHS) - lm2 * (control_2.LHS - control_2.RHS) ... - lm_k * (control_k.LHS - control_k.RHS)`

Next, the derivative of this Lagrangian is taken with respect to all control variables and all lagrange multipliers. Derivatives are are computed "though time" using `TimeAwareSymbols`, an extension of a normal Sympy symbol. For a control variable x, the total derivative over time is built up as `dL[]/dx[] + beta * dL[+1]/dx + beta * beta * dL[+2]/dx[] ...`.

The result of this unrolling and taking derivatives process are the first order conditions (FoC). All model FoCs, along with objectives, constraints, and identities, are saved into the system of equations that represents the model.

## Excluding equations from the final system

Sometimes, the user might want to exclude a constraint from the final system of equations. A common example is the household budget constraint. This is typically not included in a DSGE model. Instead, a total economic budget constraint, `Y[] = C[] + I[];` is added. Due to Walras' Law, however, these two constraints are redundant. To exclude the household budget constraint, add `@exclude` directly above it:

```
block HOUSEHOLD
{
    # ... objective, definition, and controls as above

    constraints
    {
        # Use the household budget constraint to solve the optimization, but then discard it
        # Note that lambda[] is still created and added to the model variables!
        @exclude
        C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];

        # Keep the law of motion of capital as normal
        K[] = (1 - delta) * K[-1] + I[] : q[];
    };

    # ... calibration as above
};

block EQULIBRIUM
{
    identities
    {
        Y[] = C[] + I[];
    };
};

```



## Steady State

After finding FoCs, the system will be ready to find a steady state and solve for a first-order linear approximation. To help the process, the user can write a `STEADY_STATE` block. This is a special reserved keyword block that can be placed anywhere in the GCN file. It should contain only `defintions` and `identities` as components. Here is an example of a steady state block for the RBC model:

```
block STEADY_STATE
{
    definitions
    {
        # The capital-labor ratio doesn't appear in the model, but it's very useful for solving the steady state,
        # so we can include it as a definition here.
        K_to_L[ss] = (alpha * A[ss] * P[ss] / r[ss]) ^ (1 / (1 - alpha));
    };

    identities
    {
        A[ss] = 1;
        P[ss] = 1;
        r[ss] = (1 / beta - (1 - delta));
        w[ss] = (1 - alpha) * A[ss] * mc[ss] * K_to_L[ss] ^ alpha;
        L[ss] = ((A[ss] * K_to_L[ss] ^ alpha - delta * K_to_L[ss]) ^ (-sigma_C)
                    * w[ss] / Theta
                ) ^ (1 / (sigma_C + sigma_L));

        K[ss] = K_to_L[ss] * L[ss];
        Y[ss] = A[ss] * K[ss] ^ alpha * L[ss] ^ (1 - alpha);

        I[ss] = delta * K[ss];
        C[ss] = Y[ss] - I[ss];

        U[ss] = (1 / (1 - beta)) * (C[ss] ^ (1 - sigma_C) / (1 - sigma_C) - L[ss] ^ (1 + sigma_L) / (1 + sigma_L));
        lambda[ss] = C[ss] ^ (-sigma_C) / P[ss];
        q[ss] = lambda[ss];
        TC[ss] = -(r[ss] * K[ss] + w[ss] * L[ss]);
    };
};
```

All information from the model block, including parameters and variables, are available to use in the `STEADY_STATE` block regardless of where they appear relative to each other (you can put the STEADY_STATE block at the top if you wish -- the file is not parsed top-to-bottom).

Note that these equations are not checked in any way -- if you put something in the `STEADY_STATE` block, it is taken as the Word of God, and model solving proceeds from there. If you are having trouble finding a steady state, be sure to double check these equations.

Finally, you **do not** have to provide the complete steady state system! You can include only equations, and the rest will be passed to an optimizer to be solved.

## Solving the model

Once a GCN file is written, using gEcon to do analysis is easy, as this code block shows:

```python
import gEconpy

file_path = 'GCN Files/RBC.gcn'
model = gEconpy.model_from_gcn(file_path, verbose=True)
```

When the model is loaded, you will get a message about the number of equations and variables, as well as some other basic model descriptions. You can then solve for the stead state:
```python
model.steady_state();
# Steady state found! Sum of squared residuals is 2.9196536232567403e-19
```

And get the linearized state space representation

```python
model.solve_model();
# Solution found, sum of squared residuals:  7.075155451456433e-30
# Norm of deterministic part: 0.000000000
# Norm of stochastic part:    0.000000000
```

To see how to do simulations, IRFs, and compute moments, see the example notebooks in the [example gallery](https://geconpy.readthedocs.io/en/latest/examples/gallery.html).

# Other Features

## Dynare Code Generation

Since Dynare is still the gold standard in DSGE modeling, and this is a wacky open source package written by a literally who?, gEconpy has the ability to automatically convert a solved model into a Dynare mod file. This is done as follows:

```python
print(ge.make_mod_file(model))
```

Output:
```
var A, C, I, K, L, Y, lambda, r, w;

varexo epsilon_A;

parameters alpha, beta, delta, rho_A, sigma_C, sigma_L;

alpha = 0.350;
beta = 0.990;
delta = 0.020;
rho_A = 0.950;
sigma_C = 1.500;
sigma_L = 2.000;


model;

-C - I + K(-1) * r + L * w;
I - K + K(-1) * (1 - delta);
-lambda + C ^ (-sigma_C);
-L ^ sigma_L + lambda * w;
beta * (lambda(+1) * r(+1) - lambda(+1) * (delta - 1)) - lambda;
A * K(-1) ^ alpha * L ^ (1 - alpha) - Y;
alpha * A * K(-1) ^ (alpha - 1) * L ^ (1 - alpha) - r;
A * K(-1) ^ alpha * L ^ (-alpha) * (1 - alpha) - w;
rho_A * log(A(-1)) + epsilon_A - log(A);

end;

steady_state_model;
x0 = 1 - alpha;
x1 = -delta + 1 - 1 / beta;
x2 = -x1;
x3 = alpha / x2;
x4 = x3 ^ (alpha / x0);
x5 = x0 * x4;
x6 = 1 / sigma_C;
x7 = 1 / (sigma_C + sigma_L);
x8 = (x2 / (-alpha * delta - x1)) ^ (sigma_C * x7) * (x4 ^ sigma_L * x5) ^ x7;
x9 = x8 ^ (-sigma_L * x6) * (x0 ^ (-sigma_L) * x5 ^ (sigma_L + 1)) ^ x6;
x10 = x3 * x8;


A = 1.00000000000000;
C = x9;
I = delta * x10;
K = x10;
L = x8 / x4;
Y = x8;
lambda = x9 ^ (-sigma_C);
r = x2;
w = x5;


end;

steady;
resid;

check(qz_zero_threshold=1e-20);

shocks;
var epsilon_A;
stderr 0.01;

end;

stoch_simul(order=1, irf=100, qz_zero_threshold=1e-20);
```

Since this model included a `STEADY_STATE` block with a complete solution, a `steady_state_model` was generated. If no steady state equations are provided, or if the provided solution is incomplete, an `initvals` block will be generated instead, and jittered steady-state values found by gEconpy will be used as inital values.

### Warings about Dynare Code
* If your model includes calibrated equations, the generated Dynare code **will not** work out of the box. You need to analyically compute the steady state values and add a deterministic relationship (that beings with `#`) to the model block.
* Currently, priors are not generated in the Dynare code. You will need to add these manually.

## Estimation

Bayesian estimation of a model can be done using PyMC. Currently, only models with a fully analytic steady state are supported. To sample a model, create a `PyMCStateSpace` using `ge.statespace_from_gcn`. For a complete example, see the estimation example notebooks in the [example gallery](https://geconpy.readthedocs.io/en/latest/examples/gallery.html).

```python
import gEconpy as ge
file_path = 'GCN Files/RBC.gcn'
ss_mod = ge.statespace_from_gcn(file_path)
```

Output:
```
Model Building Complete.
Found:
	9 equations
	9 variables
		The following variables were eliminated at user request:
			TC_t, U_t
		The following "variables" were defined as constants and have been substituted away:
			mc_t
	1 stochastic shock
		 0 / 1 has a defined prior.
	6 parameters
		 6 / 6 parameters has a defined prior.
	0 parameters to calibrate.
Model appears well defined and ready to proceed to solving.

Statespace model construction complete, but call the .configure method to finalize.
```

As prompted, you will need to call the `.configure` method. This is where you can choose which model variables are observed, which have measurement error, and whether to model a full shock covariance matrix, or only the main diagonal:

```python
ss_mod.configure(
    observed_states=["Y"],
    measurement_error=None,
    full_shock_covaraince=False,
    solver="scan_cycle_reduction",
    mode="JAX",
    use_adjoint_gradients=True,
    max_iter=20,
)
```

The output of `.configure` will be a table of variables, telling you what names to use in a PyMC model block. For full details about the PyMCStateSpace API, have a look at the documentation in pymc-extras:

```
                  Model Requirements

  Variable          Shape   Constraints   Dimensions
 ────────────────────────────────────────────────────
  alpha             ()                          None
  beta              ()                          None
  delta             ()                          None
  rho_A             ()                          None
  sigma_C           ()      Positive            None
  sigma_L           ()      Positive            None
  sigma_epsilon_A   ()      Positive            None

 These parameters should be assigned priors inside a
         PyMC model block before calling the
            build_statespace_graph method.
```

From here, create a PyMC model and assign priors. These can be absolutely anything:

```python
import pymc as pm
import numpy as np

with pm.Model(coords=ss_mod.coords) as pm_mod:
    alpha = pm.Beta("alpha", alpha=2, beta=10)
    beta = pm.Beta("beta", alpha=2, beta=30)
    delta = pm.Beta("delta", alpha=1, beta=30)
    rho_A = pm.Beta("rho_A", alpha=10, beta=2)
    sigma_C = pm.TruncatedNormal("sigma_C", mu=1.5, sigma=2, lower=1.01, upper=np.inf)
    sigma_L = pm.TruncatedNormal("sigma_L", mu=1.5, sigma=2, lower=1, upper=np.inf)
    sigma_epsilon_A = pm.Exponential("sigma_epsilon_A", 10)

    ss_mod.build_statespace_graph(
        data,
        add_norm_check=True,
        add_solver_success_check=True,
        add_steady_state_penalty=True,
    )
```

From here, you can estimate your model using a wide number of algorithms, including Maximum a Posteriori (MAP), Metropolis-Hastings, Hamiltonian Monte Carlo (with NUTS), or even normalizing-flow augmented NUTS. For details, see the estimation example notebooks in the [example gallery](https://geconpy.readthedocs.io/en/latest/examples/gallery.html), as well as PyMC documentation.
