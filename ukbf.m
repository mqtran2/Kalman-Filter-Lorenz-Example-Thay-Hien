function out = ukbf(model,obser,data,time,R,Q,x0,P0,q)

% 
%
% Thanks to Adam Attarian for help in some of the code
% Thanks to Hien Tran for his patience
%
% implementation was done using Sarrka(2007)IEEE
%
%UKBF Unscented Kalman Bucy Filter
%
%  this implements a continuous-discrete unscented kalman filter 
%     INPUTS:
%          model: rhs of ODE system (include values for parameters)
%          obser: general function for observations (z = h(x))
%          data: data points used for filter (column vectors)
%          time: time period observations occur over
%          R: covariance noise for data, constant
%          Q: covariance noise for process, constant
%          x0: initial condition for model
%          P0: initial condition for covariance
%          q: parameters
%     OUTPUTS:
%          out.xfilter: filter output
%          out.time: time scale
%          out.P: covariance matrices for each time
%          out.sd: +/- 2 standard deviations

%initialize the variables
N = length(data);
L = numel(x0);
xc = zeros(L,N);
pc = zeros(L,L,N);
sd = zeros(N,L);

%assign the initial conditions to the respective variables
xc(:,1) = x0;
P(:,:,1) = P0;
sd(1,:) = (sqrt(diag(P(:,:,1)))*2)';

for i = 2:N;
    %calculate the Unscented Transform for the model
    [mk, Pk, Pkc] = UT(model,xc(:,i-1),pc(:,:,i-1),[time(i-1), time(i)],q);
    Pkp = Pk + Q;
    %calculate the Unscented Transform for the data
    [zk, Sk, Skc] = UTdata(obser,mk,Pkp);
    Sk = Sk+R;
    %calculate the correction step (instead of using inverse function)
    LL = chol(Sk);
    U = Skc/LL;
    
    pc(:,:,i) = Pkp - U*U';
    xc(:,i) = mk + U*(LL'\(data(i,:)'-zk));
    
    sd(i,:) = (sqrt(diag(pc(:,:,i)))*2)';
end

out.xfilter=xc';
out.time = time;
out.P = pc;
out.sd = sd;

function [xt, Pt, Ptc] = UT(f,x,p,time,q,alpha,beta,kappa)
% unscented transformation for model
%
%Input:
%     f: nonlinear map
%     x: state estimate reference point
%     p: covariance
%     time: vector with times points to solve over
%     alpha: tuning parameter (default = 1)
%     beta: tuning parameter  (default = 2)
%     kappa: tuning parameter (default = 0)
%
%Output:
%    xt: transformed mean
%    Pt: transformed covariance
%    Ptc: transformed deviations

if (nargin == 5)
    alpha = 1;
    beta = 2;
    kappa = 3-numel(x);
end

n = numel(x);
lambda = alpha^2*(n+kappa)-n;

%%%%%%%%% calculates matrix square root
%%%%%%%%% c: matrix squre root of P => p = c*c'
if any(eig(p)<0)
    c = real(sqrt(p));
else
     %c = chol(p)';
     [sm, lam, xn] = svd(p);
     sqrtD = sqrt(lam);
     c = sm*sqrtD*sm';
end

%%%%%%%% paran: = sqrt(n+lambda)
den = n+lambda;
paran = sqrt(den);
rootP = c*paran;

% calculates sigma points, Xi = xhat +/- paran*c
Y = x(:,ones(1,n));
Xi = [x Y+rootP Y-rootP];

% calculating the weights of the mean and covariance
Wm = [lambda/den 0.5/den+zeros(1,2*n)];
Wc = Wm;
Wc(1) = (lambda/(den+1-alpha^2+beta));
wm = [Wm(ones(1,length(Wm)),:)'];

%calculating the statistics of the nonlinear transformation, Y
% and the transformed mean, xt
L = size(Xi,2);
Y = zeros(n,L);
xt = zeros(n,1);
Pt = zeros(n);
Ptc = Pt;
W = (eye(L)-wm)*diag(Wc)*(eye(L)-wm)';

%runs the sigma points through the nonlinear transformation, Y
%and calculates the transformed mean 
options = odeset('RelTol',1e-6,'Abstol',1e-6);
for u = 1:L;
    [td,xd] = ode15s(f,time,Xi(:,u),[],q);
    Yi(:,u) = xd(end,:);
    %calculates the transformed mean, xt
    xt = xt + Yi(:,u)*Wm(u)';
end

%calculates the transformed covariance
Pt = Yi*W*Yi';
Ptc = Xi*W*Yi';

end %terminates UT function

function [xt, Pt, Ptc] = UTdata(h,x,p,alpha,beta,kappa)
% unscented transformation for data
%
%Input:
%     h: observation function
%     x: state estimate reference point
%     p: covariance
%
%Output:
%    xt: transformed mean
%    Pt: transformed covariance
%    Ptc: transformed deviations

if (nargin == 3)
    alpha = 1;
    beta = 2;
    kappa = 3-numel(x);
end

n = numel(x);
lambda = alpha^2*(n+kappa)-n;
% calculates matrix square root
% c: matrix squre root of P => p = c*c'
c = chol(p)';


den = n+lambda;
paran = sqrt(den);
rootP = c*paran;

% calculates sigma points, Xi = xhat +/- paran*c
Y = x(:,ones(1,n));
Xi = [x Y+rootP Y-rootP];

% calculating the weights of the mean and covariance
Wm = [lambda/den 0.5/den+zeros(1,2*n)];
Wc = Wm;
Wc(1) = lambda/(den+(1-alpha^2+beta));
wm = [Wm(ones(1,length(Wm)),:)'];

%calculating the statistics of the nonlinear transformation, Y
% and the transformed mean, xt
L = size(Xi,2);
Y = zeros(size(h,1),L);
xt = zeros(n,1);
Pt = zeros(n);
Ptc = Pt;

for j = 1:L;
    Yi(:,j) = feval(h,Xi(:,j));
end
xt = Yi*Wm';
W = (eye(L)-wm)*diag(Wc)*(eye(L)-wm)';
%calculates the transformed covariance
Pt = Yi*W*Yi';
Ptc = Xi*W*Yi';

%end of function; UT
end

%terminates Unscented kalman bucy filter
end


