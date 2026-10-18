var A, C, I, K, L, TC, U, Y, lambda, q, r, w;

varexo epsilon_A;

parameters alpha, beta, delta, rho_A, sigma_C, sigma_L;

alpha = 0.350;
beta = 0.985;
delta = 0.025;
rho_A = 0.950;
sigma_C = 2.000;
sigma_L = 1.500;


model;

-C - I + K(-1)*r + L*w;
I - K + K(-1)*(1 - delta);
beta*U(+1) + C^(1 - sigma_C)/(1 - sigma_C) - L^(sigma_L + 1)/(sigma_L + 1) - U;
beta*(lambda(+1)*r(+1) - q(+1)*(delta - 1)) - q;
-lambda + C^(-sigma_C);
-L^sigma_L + lambda*w;
-lambda + q;
rho_A*log(A(-1)) + epsilon_A - log(A);
A*K(-1)^alpha*L^(1 - alpha) - Y;
-K(-1)*r - L*w - TC;
alpha*A*K(-1)^(alpha - 1)*L^(1 - alpha) - r;
A*K(-1)^alpha*L^(-alpha)*(1 - alpha) - w;

end;

steady_state_model;
A = 1.00000000000000;
C = (((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^(sigma_L + 1)*(1 - alpha)^(-sigma_L))^(1./sigma_C)*(((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L)))^((-sigma_L)/sigma_C);
I = alpha*delta*((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L))/(-(1 - delta) + 1/beta);
K = alpha*((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L))/(-(1 - delta) + 1/beta);
L = (alpha/(-(1 - delta) + 1/beta))^(-alpha/(1 - alpha))*((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L));
TC = -alpha*((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L)) - ((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L))*(1 - alpha);
U = (((((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^(sigma_L + 1)*(1 - alpha)^(-sigma_L))^(1./sigma_C)*(((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(1 - sigma_C)/(1 - sigma_C) - ((alpha/(-(1 - delta) + 1/beta))^(-alpha/(1 - alpha))*((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L)))^(sigma_L + 1)/(sigma_L + 1))/(1 - beta);
Y = ((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L));
lambda = ((((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^(sigma_L + 1)*(1 - alpha)^(-sigma_L))^(1./sigma_C)*(((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C);
q = ((((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha))^(sigma_L + 1)*(1 - alpha)^(-sigma_L))^(1./sigma_C)*(((-(1 - delta) + 1/beta)/(-alpha*delta - (1 - delta) + 1./beta))^(sigma_C/(sigma_C + sigma_L))*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha)*((alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha)))^sigma_L)^(1./(sigma_C + sigma_L)))^((-sigma_L)/sigma_C))^(-sigma_C);
r = -(1 - delta) + 1/beta;
w = (alpha/(-(1 - delta) + 1/beta))^(alpha/(1 - alpha))*(1 - alpha);


end;

steady;
resid;

check(qz_zero_threshold=1e-20);

shocks;
var epsilon_A;
stderr 0.01;

end;

stoch_simul(order=2, irf=0, periods=0, pruning, qz_zero_threshold=1e-20);
