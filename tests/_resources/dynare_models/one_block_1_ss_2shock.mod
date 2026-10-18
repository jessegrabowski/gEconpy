var A, B, C, I, K, L, U, Y, lambda, q;

varexo epsilon_A, epsilon_B;

parameters alpha, beta, delta, rho_A, rho_B, tau, theta;

alpha = 0.350;
beta = 0.990;
delta = 0.020;
rho_A = 0.950;
rho_B = 0.950;
tau = 2.000;
theta = 0.357;


model;

rho_A * log(A(-1)) + epsilon_A - log(A);
rho_B * log(B(-1)) + epsilon_B - log(B);
A * K(-1) ^ alpha * L ^ (1 - alpha) - Y;
-C - I + Y;
I - K + K(-1) * (1 - delta);
beta * U(+1) + B * (C ^ theta * (1 - L) ^ (1 - theta)) ^ (1 - tau) / (1 - tau) - U;
theta * B * (C ^ theta * (1 - L) ^ (1 - theta)) ^ (1 - tau) / C - lambda;
A * K(-1) ^ alpha * L ^ (-alpha) * lambda * (1 - alpha) + B * (C ^ theta * (1 - L) ^ (1 - theta)) ^ (1 - tau) * (theta - 1) / (1 - L);
-lambda + q;
beta * (alpha * A(+1) * K ^ (alpha - 1) * L(+1) ^ (1 - alpha) * lambda(+1) - q(+1) * (delta - 1)) - q;

end;

steady_state_model;
x0 = alpha - 1;
x1 = -x0;
x2 = theta * x1;
x3 = alpha * theta - 1;
x4 = x2 * (-beta * (1 - delta) + 1) / (-beta * (-delta * x1 - x3) - x3);
x5 = theta - 1;
x6 = x4 * (-(-x2 - x3 * x4) / (delta * x4 * x5)) ^ (1 / x1);
x7 = delta * x6;
x8 = x4 ^ (-x0);
x9 = x6 ^ alpha;
x10 = -x7 + x8 * x9;
x11 = tau - 1;
x12 = (x10 ^ theta * (1 - x4) ^ (-x5)) ^ (-x11);
x13 = theta * x12 / x10;


A = 1.00000000000000;
B = 1.00000000000000;
C = x10;
I = x7;
K = x6;
L = x4;
U = -x12 / (x11 * (1 - beta));
Y = x8 * x9;
lambda = x13;
q = x13;


end;

steady;
resid;

check(qz_zero_threshold=1e-20);

shocks;
var epsilon_A;
stderr 0.01;

var epsilon_B;
stderr 0.02;

end;

stoch_simul(order=2, irf=0, periods=0, pruning, qz_zero_threshold=1e-20);
