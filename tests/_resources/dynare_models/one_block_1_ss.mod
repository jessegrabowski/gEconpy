var A, C, I, K, L, U, Y, lambda, q;

varexo epsilon;

parameters alpha, beta, delta, rho, tau, theta;

alpha = 0.350;
beta = 0.990;
delta = 0.020;
rho = 0.950;
tau = 2.000;
theta = 0.357;


model;

rho*log(A(-1)) + epsilon - log(A);
A*K(-1)^alpha*L^(1 - alpha) - Y;
-C - I + Y;
I - K + K(-1)*(1 - delta);
beta*U(+1) - U + (C^theta*(1 - L)^(1 - theta))^(1 - tau)/(1 - tau);
theta*(C^theta*(1 - L)^(1 - theta))^(1 - tau)/C - lambda;
A*K(-1)^alpha*L^(-alpha)*lambda*(1 - alpha) + (C^theta*(1 - L)^(1 - theta))^(1 - tau)*(theta - 1)/(1 - L);
-lambda + q;
beta*(alpha*A(+1)*K^(alpha - 1)*L(+1)^(1 - alpha)*lambda(+1) - q(+1)*(delta - 1)) - q;

end;

steady_state_model;
x0 = alpha - 1;
x1 = -x0;
x2 = 1 / x1;
x3 = theta - 1;
x4 = -beta * (1 - delta) + 1;
x5 = delta * x1;
x6 = alpha * theta - 1;
x7 = -beta * (-x5 - x6) - x6;
x8 = theta * x1;
x9 = x4 / x7;
x10 = x8 * x9;
x11 = (-x2 * x7 * (-x10 * x6 - x8) / (delta * theta * x3 * x4)) ^ x2;
x12 = theta * x11 * x5 * x9;
x13 = x10 ^ (-x0);
x14 = x10 * x11;
x15 = x14 ^ alpha;
x16 = -x12 + x13 * x15;
x17 = tau - 1;
x18 = (x16 ^ theta * (1 - x10) ^ (-x3)) ^ (-x17);
x19 = theta * x18 / x16;


A = 1.00000000000000;
C = x16;
I = x12;
K = x14;
L = x10;
U = -x18 / (x17 * (1 - beta));
Y = x13 * x15;
lambda = x19;
q = x19;


end;

steady;
resid;

check(qz_zero_threshold=1e-20);

shocks;
var epsilon;
stderr 0.01;

end;

stoch_simul(order=2, irf=0, periods=0, pruning, qz_zero_threshold=1e-20);
