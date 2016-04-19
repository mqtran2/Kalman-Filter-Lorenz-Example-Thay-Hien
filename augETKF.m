function out=augETKF(dynfun,obsfun,data,time,x0,R,Vp,P0x,q0,P0q,Vq,N,param,r)
%
% augETKF: Augmented Ensemble Transform Kalman Filter
%
%      This implements the square root ensemble kalman filter using the
%      transform matrix given by Bishop (2001).  This implementation uses a
%      nonlinear observation function coupled with variance inflation to
%      estimate both the states and parameters in the joint sense.
%
%      INPUTS:
%           dynfun: rhs of ODE system (model) including the parameters (column vector)
%           obsfun: nonlinear observation function
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           x0: initial condition for system
%           R:  Observation noise covariance, constant
%           Vp:  Process noise covariance
%           P0x: initial state filter covariance
%		    q0: initial parameter values
%           P0q: initial parameter filter covariance
%           Vq: Process noise covariance for parameters
%           N: number of ensembles (particles)
%           param: index of parameters to estimate
%           r: covariance inflation parameter
%       OUTPUTS:
%           out.xfilter: state filter output
%           out.qfilter: filter of parameter estimates (last value is final
%                        estimate of parameter; best fit) 
%           out.Px: State covariance matrices for each time
%           out.Pq: Parameter estimate covariance matrices for each time
%           out.time: time scale
%           out.data: original data used for filter
%           out.tsdq: +/- 3 std. deviations of parameter estimate
%           out.tsdx: +/- 3 std. deviations of state filter
%
%   
%
%   Implementation was done using both the notes from John Harlim and
%   Evensen (2004)
%
%   this code comes with no guarantee of any kind

if ischar(dynfun)==1; dynfun=str2func(dynfun); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end
if (nargin == 13); r = 0.05; end

%initialization 
x0=x0(:); q0 = q0(:); 
L = length(data);   % number of time iterations
dt = time(2)-time(1);   % time step size
n = numel(x0);      % n, Dimension of state
m = numel(q0(param));      % m, dimension of parameters
Ap = zeros(n+length(q0),N);

%setting up parameter ensemble
Q = ones(length(q0),N);
param_const = setxor(1:length(q0),param);
Q(param_const,:) = repmat(q0(param_const),1,N);


% initial ensembles, Ax, model, Q, parameters
%Ax = repmat(x0,1,N) + V*randn(n,N);
Ax = repmat(x0,1,N) + sqrt(P0x)*randn(n,N);
%Q(param,:) = repmat(q0(param),1,N) + Vq*randn(m,N);
Q(param,:) = repmat(q0(param),1,N) + sqrt(P0q)*randn(m,N);

%setting up actual ensemble matrix, A
A = [Ax; Q];

%initialize filter 
xfilter=zeros(n,L);
qfilter=zeros(m,L);
Pq=zeros(m,m,L);
Px=zeros(n,n,L);
tsdx=zeros(L,n);
tsdq=zeros(L,m);

%initial conditions
xfilter(:,1) = x0;
qfilter(:,1) = q0(param);
Pq(:,:,1) = P0q;
Px(:,:,1) = P0x;
tsdx(1,:) = (sqrt(diag(Px(:,:,1))))';
tsdq(1,:) = (sqrt(diag(Pq(:,:,1))))';

invR = inv(R);
%main filter loop
options = odeset('RelTol',1e-5,'AbsTol',1e-5);
for k=2:L
    % Prediction Step: 
    % push each 'particle' of the ensemble through the model
    for j=1:N
        Ap(1:n,j) = rk4(dynfun,dt,time(k-1),A(1:n,j),A(n+1:end,j));
        %[tless, Asol] = ode15s(dynfun,[time(k-1) time(k)],A(1:n,j),options,A(n+1:end,j));
        %Ap(1:n,j) = Asol(end,:);
    end
    Ap(1:n,:) = Ap(1:n,:) + sqrt(Vp)*randn(n,N);
    Ap(n+param,:) = A(n+param,:) + sqrt(Vq)*randn(m,N);
    Ap(n+param_const,:) = A(n+param_const,:);
    %Ap(n+param,:) = A(n+1:end,:);
    
    % Analysis Step:
    % update the estimate of the state given the obervation
    % ie, calculate posterior through likelihood function
    
    %nonlinear observation function
    for j=1:N;
        G(:,j) = feval(obsfun,Ap(1:n,j));
    end
    
    %calculate V, the nonlinear observation matrix
    I2 = 1/N*ones(N,N);
    Vbar = G*I2;
    V = G-Vbar;
    
    %calculate ensemble perturbation matrix (51)
    I = 1/N*ones(N,N);
    Abar = Ap*I;
    Aprime = Ap-Abar;
    
    %variance inflation step
    Aprime = sqrt(1+r)*Aprime;
    V = sqrt(1+r)*V;
    
    %calculate SVD of J
    J = ((N-1)/(1+r))*eye(N) + V'*invR*V;
    [X Gamma Psi] = svd(J,'econ');
    
    %kalman gain, K
    K = Aprime*pinv(J)*V'*invR;
    
    %update equations
    % posterior mean, u_m+1
    u_a = Abar(:,1) + K*(data(k,:)'-feval(obsfun,Abar(:,1)));
    
    %compute transformation matrix, T
    T = sqrt(N-1)*X*chol(inv(Gamma))*X';
    
    %posterior perturbation matrix of ensembles
    U = Aprime*T;
    
    %Posterior ensemble
    u_mean = repmat(u_a,1,N);
    A = u_mean + U;
 
    %find mean and covariance of updated ensemble for filter
    meanAx = mean(A(1:n,:),2);
    covAx = (A(1:n,:)-meanAx(:,ones(1,N)))*(A(1:n,:)-meanAx(:,ones(1,N)))'/(N-1);
    meanAq = mean(A(n+param,:),2);
    covAq = (A(n+param,:)-meanAq(:,ones(1,N)))*(A(n+param,:)-meanAq(:,ones(1,N)))'/(N-1);

    
    %store solutions
    xfilter(:,k) = meanAx;
    qfilter(:,k) = meanAq;
    Px(:,:,k) = covAx;
    Pq(:,:,k) = covAq;
    tsdx(k,:) = (sqrt(diag(covAx)))';
    tsdq(k,:) = (sqrt(diag(covAq)))';
    ensemble(:,:,k) = A;

end

out.xfilter=xfilter';
out.qfilter=qfilter';
out.Px=Px;
out.Pq=Pq;
out.ensemble = ensemble;
out.time=time;
out.data=data;
out.tsdx=tsdx;
out.tsdq=tsdq;

function dx=rk4(rhs,h,t,x,q)

% fourth order explicit rk integrator

k1=feval(rhs,t,x,q);
k2=feval(rhs,t+h/2,x+h/2*k1,q);
k3=feval(rhs,t+h/2,x+h/2*k2,q);
k4=feval(rhs,t+h,x+h*k3,q);

dx=x+(h/6)*(k1+2*k2+2*k3+k4);
