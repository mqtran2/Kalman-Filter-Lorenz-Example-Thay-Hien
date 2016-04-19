function out=augEnKF(dynfun,M,data,time,x0,R,V,P0x,q0,P0q,Vq,N,param)
%
% augEnKF: Augmented Ensemble Kalman Filter
%
%      This implements the ensemble kalman filter using the method from
%      Evensen (2004). This implementation uses linear observations given by
%      the observation matrix operator, M.  This EnKF code does dual
%      estimation of both the states and parameters through joint
%      estimation techniques.
%
%      INPUTS:
%           dynfun: rhs of ODE system (model) including the parameters (column vector)
%           M: operator matrix for observations
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           x0: initial condition for system
%           R:  Observation noise covariance, constant
%           V:  Process noise covariance
%           P0x: initial state filter covariance
%		    q0: initial parameter values
%           P0q: initial parameter filter covariance
%           Rq: Observation noise covariance for parameters
%           Vq: Process noise covariance for parameters
%           N: number of ensembles (particles)
%           param: index of parameters to estimate
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
%

if ischar(dynfun)==1; dynfun=str2func(dynfun); end
if ischar(M)==1; M=str2func(M); end

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
Ax = repmat(x0,1,N) + sqrt(P0x)*randn(n,N);
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

%main filter loop
options = odeset('RelTol',1e-6,'AbsTol',1e-6);
for k=2:L
    % Prediction Step: 
    % push each 'particle' of the ensemble through the model
    for j=1:N
        Ap(1:n,j) = rk4(dynfun,dt,time(k-1),A(1:n,j),A(n+1:end,j));
%         [tless, Asol] = ode15s(dynfun,[time(k-1) time(k)],A(1:n,j),options,A(n+1:end,j));
%         Ap(1:n,j) = Asol(end,:);
    end
    Ap(1:n,:) = Ap(1:n,:) + sqrt(V)*randn(n,N);
    Ap(n+param,:) = A(n+param,:) + sqrt(Vq)*randn(m,N);
    Ap(n+param_const,:) = A(n+param_const,:);
    %Ap(n+1:end,:) = A(n+1:end,:);
    
    % Analysis Step:
    % update the estimate of the state given the obervation
    % ie, calculate posterior through likelihood function
    
    %calculate ensemble perturbation matrix (51)
    I = 1/N*ones(N,N);
    Abar = Ap*I;
    Aprime = Ap-Abar;
    
    %calculate ensemble covariance matrix, C_psipsi (52)
    %C_psipsi = (1/(N-1))*Aprime*Aprime';
    
    %calculate measurement matrix (54)
    D = repmat(data(k,:)',1,N);
    
    %calculate measurement perturbation matrix (55)
    E = R*randn(length(data(k,:)'),N);
    
    %calculate measurement error covariance matrix, C_ee (56)
    C_ee = (1/(N-1))*E*E';
    
    % calculate matrix holding measurements of ensemble perturbations
    % and other matrices required for update equation
    % (58,60,61,63)
    Dprime = D-M*A;
    S = M*Aprime;
    C = S*S' + (N-1)*C_ee;
    X = eye(N) + S'*(C\Dprime);
    
    %Update equation, A^a (62)
    A = Ap*X;
 
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
