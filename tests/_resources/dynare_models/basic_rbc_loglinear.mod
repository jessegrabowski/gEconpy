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

-C - I + K(-1)*r + L*w;
I - K + K(-1)*(1 - delta);
-lambda + C^(-sigma_C);
-L^sigma_L + lambda*w;
beta*(lambda(+1)*r(+1) - lambda(+1)*(delta - 1)) - lambda;
A*K(-1)^alpha*L^(1 - alpha) - Y;
alpha*A*K(-1)^(alpha - 1)*L^(1 - alpha) - r;
A*K(-1)^alpha*L^(-alpha)*(1 - alpha) - w;
rho_A*log(A(-1)) + epsilon_A - log(A);

end;

initval;
A = 1.000;
C = 2.358;
I = 0.715;
K = 35.732;
L = 0.820;
Y = 3.073;
lambda = 0.276;
r = 0.030;
w = 2.436;

end;

steady(tolf=1e-14, tolx=1e-14);
resid;

check(qz_zero_threshold=1e-20);

shocks;
var epsilon_A;
stderr 0.01;

end;

stoch_simul(order=1, irf=0, periods=0, qz_zero_threshold=1e-20, loglinear);
