var C, Div, I, K, L, LHS, LHS_w, RHS, RHS_w, TC, U, Y, lambda, mc, pi, pi_obj, pi_star, q, r, r_G;
var shock_preference, shock_technology, w, w_star;

varexo epsilon_R, epsilon_Y, epsilon_pi, epsilon_preference;

parameters alpha, beta, delta, eta_p, eta_w, gamma_I, gamma_R, gamma_Y, gamma_pi, phi_H, phi_pi_obj;
parameters psi_p, psi_w, rho_pi_dot, rho_preference, rho_technology, sigma_C, sigma_L;

alpha = 0.350;
beta = 0.990;
delta = 0.025;
eta_p = 0.750;
eta_w = 0.750;
gamma_I = 10.000;
gamma_R = 0.900;
gamma_Y = 0.050;
gamma_pi = 1.500;
phi_H = 0.500;
phi_pi_obj = 1.000;
psi_p = 0.600;
psi_w = 0.782;
rho_pi_dot = 0.924;
rho_preference = 0.950;
rho_technology = 0.950;
sigma_C = 2.000;
sigma_L = 1.500;


model;

#pi_ss = 1.00000000000000;
#r_G_ss = 1/beta;
-C + Div - I + K(-1)*r + L*w;
I*(1 - gamma_I*(I/I(-1) - 1)^2/2) - K + K(-1)*(1 - delta);
beta*U(+1) - U + shock_preference*(-L^(sigma_L + 1)/(sigma_L + 1) + (-phi_H*C(-1) + C)^(1 - sigma_C)/(1 - sigma_C));
-beta*phi_H*shock_preference(+1)*(-phi_H*C + C(+1))^(-sigma_C) - lambda + shock_preference*(-phi_H*C(-1) + C)^(-sigma_C);
beta*gamma_I*I(+1)^2*q(+1)*(-1 + I(+1)/I)/I^2 - lambda - q*(gamma_I*I*(I/I(-1) - 1)/I(-1) + gamma_I*(I/I(-1) - 1)^2/2 - 1);
beta*(lambda(+1)*r(+1) - q(+1)*(delta - 1)) - q;
beta*lambda(+1)/pi(+1) - lambda/r_G;
-LHS_w + RHS_w;
beta*eta_w*LHS_w(+1)*pi(+1)*(w_star(+1)/w_star)^(1/psi_w) - LHS_w + L*lambda*w_star*(w/w_star)^((psi_w + 1)/psi_w)/(psi_w + 1);
beta*eta_w*RHS_w(+1)*(pi(+1)*w_star(+1)/w_star)^((psi_w + 1)*(sigma_L + 1)/psi_w) - RHS_w + shock_preference*(L*(w/w_star)^((psi_w + 1)/psi_w))^(sigma_L + 1);
eta_w*(pi*w/w(-1))^(1/psi_w) + (w/w_star)^(1/psi_w)*(1 - eta_w) - 1;
rho_preference*log(shock_preference(-1)) + epsilon_preference - log(shock_preference);
-Div + TC + Y;
K(-1)^alpha*L^(1 - alpha)*shock_technology - Y;
-K(-1)*r - L*w - TC;
alpha*K(-1)^(alpha - 1)*L^(1 - alpha)*mc*shock_technology - r;
mc*shock_technology*(K(-1)/L)^alpha*(1 - alpha) - w;
rho_technology*log(shock_technology(-1)) + epsilon_Y - log(shock_technology);
-LHS + RHS*(psi_p + 1);
beta*eta_p*LHS(+1)*pi_star*pi(+1)^(1/psi_p)/pi_star(+1) - LHS + Y*lambda*pi_star;
beta*eta_p*RHS(+1)*pi(+1)^((psi_p + 1)/psi_p) - RHS + Y*lambda*mc;
eta_p*pi^(1/psi_p) - 1 + pi_star^(-1/psi_p)*(1 - eta_p);
gamma_R*log(r_G(-1)/r_G_ss) + gamma_Y*(1 - gamma_R)*log(Y/Y(-1)) + gamma_pi*(1 - gamma_R)*log(-log(pi_obj) + pi/pi_ss) + epsilon_R + (1 - gamma_R)*log(pi_obj) - log(r_G/r_G_ss);
rho_pi_dot*log(pi_obj(-1)) + epsilon_pi + (1 - rho_pi_dot)*log(phi_pi_obj) - log(pi_obj);

end;

steady_state_model;
C = ((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C)/(1 - phi_H);
Div = -alpha*((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))/(psi_p + 1) - ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))*(1 - alpha)/(psi_p + 1) + ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L));
I = alpha*delta*((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))/((psi_p + 1)*(-(1 - delta) + 1/beta));
K = alpha*((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))/((psi_p + 1)*(-(1 - delta) + 1/beta));
LHS = ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))*(((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C)*(1 - beta*phi_H)/(-beta*eta_p + 1);
LHS_w = ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))*(((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C)*(1 - alpha)*(1 - beta*phi_H)/((1 - beta*eta_w)*(psi_p + 1)*(psi_w + 1));
L = (1/(psi_p + 1))^(-1/(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(-alpha/(1 - alpha))*((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))/(psi_p + 1);
RHS = ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))*(((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C)*(1 - beta*phi_H)/((psi_p + 1)*(-beta*eta_p + 1));
RHS_w = ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))*(((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C)*(1 - alpha)*(1 - beta*phi_H)/((1 - beta*eta_w)*(psi_p + 1)*(psi_w + 1));
TC = -alpha*((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))/(psi_p + 1) - ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))*(1 - alpha)/(psi_p + 1);
U = ((((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(1 - sigma_C)/(1 - sigma_C) - ((1/(psi_p + 1))^(-1/(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(-alpha/(1 - alpha))*((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L))/(psi_p + 1))^(sigma_L + 1)/(sigma_L + 1))/(1 - beta);
Y = ((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L));
lambda = (((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C)*(1 - beta*phi_H);
mc = 1/(psi_p + 1);

pi_obj = 1.00000000000000;
pi = 1.00000000000000;
pi_star = 1.00000000000000;
q = (((1 - alpha)/(psi_p + 1))^((-sigma_L)/sigma_C)*((1 - beta*phi_H)/(psi_w + 1))^(1./sigma_C)*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/sigma_C)*(((1 - alpha)/(psi_p + 1))^(-sigma_L/(sigma_C + sigma_L))*(((-beta)*phi_H + 1)/(psi_w + 1))^(1./(sigma_C + sigma_L))*((1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^((sigma_L + 1)/(sigma_C + sigma_L))*((-(1 - delta) + 1/beta)/((1 - phi_H)*(-alpha*delta/(psi_p + 1) - (1 - delta) + 1./beta)))^(sigma_C/(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C)*(1 - beta*phi_H);
r_G = 1/beta;
r = -(1 - delta) + 1/beta;
shock_preference = 1.00000000000000;
shock_technology = 1.00000000000000;
w = (1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha);
w_star = (1/(psi_p + 1))^(1./(1 - alpha))*(alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha);


end;

steady;
resid;

check(qz_zero_threshold=1e-20);

shocks;
var epsilon_R;
stderr 0.01;

var epsilon_Y;
stderr 0.01;

var epsilon_pi;
stderr 0.01;

var epsilon_preference;
stderr 0.01;

end;

stoch_simul(order=2, irf=0, periods=0, pruning, qz_zero_threshold=1e-20);
