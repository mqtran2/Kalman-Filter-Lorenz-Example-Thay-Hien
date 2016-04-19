function out=srparckf(dynfun,obsfun,data,time,q0,Re,Rr,P0q,xstate0,param)
%
%SrParCKF: Parameter Estimation Square Root Cubature Kalman Filter
%     PARCKF(dynfun,obsfun,data,time,q0,Re,Rr,P0q,xstate0,param)
%   This implements a continuous-discrete cubature kalman bucy filter to
%   simultaneously estimate the system parameters
%   parameteres.
%      INPUTS:
%           dynfun: rhs of ODE system (model) including the parameters (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           q0: initial guess of parameter value
%           Re: Parameter Measurement noise covariance, constant
%           Rr: Parameter Process noise covariance, constant
%           P0q: initial parameter filter covariance
%           xstatex0: initial condition for model
%           param: index of parameters to estimate
%       OUTPUTS:
%           out.qfilter: filter of parameter estimates (last value is final
%                        estimate of parameter; best fit) 
%           out.Pq: Parameter estimate covariance matrices for each time
%           out.time: time scale
%           out.data: original data used for filter
%           out.tsdq: +/- 3 std. deviations of parameter estimate
%
% Brett Matzuka & Adam Attarian, NCSU, July 2009
%
% See 'Cubature Kalman Filters' by Simon Haykin for more info
% on dual estimation and the CKF
%
% This code comes with no guarantees of any kind

if ischar(dynfun)==1; dynfun=str2func(dynfun); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end

xstate0=xstate0(:); q0=q0(:); % columns, please. 
L = length(xstate0);
m=numel(q0(param));
N=length(data);
dt=time(2)-time(1);

% init the filter
Pq=cell(m,m,N);

% fill in the initial conditions for both filters.
Rr = chol(Rr)';
Re = chol(Re)';

qfilter(:,1)=q0(param);   % initial parameter filter value
Pq{1}=chol(P0q)';  % initial parameter covariance
Rk = Rr;    % initial Parameter process noise
xstate(:,1) = xstate0; % initial condition for model
tsdq(1,:) = (sqrt(diag(Pq{1}))*3)';

% initialize parameters (both those to be estimated and those held
% constant)
w = ones(length(q0),N);
w(:,1) = q0;

% main filter loop
for k=2:N;
    %time update of parameters
    qminus=qfilter(:,k-1);
    %time update of covariances 
    Pqminus=Pq{k-1}+Rk;   
    
    % Cubature points for the parameters.  
    Xq=cubaturepoints_sr(qminus,Pqminus);  %eq. 44
    
    % Changing stencil to include all parameters; Xqnew = sigma points for
    % parameters plus rows of the parameters which are held constant
    Xqnew = ones(length(q0),2*m);
    Xqnew(param,:) = Xq(:,:);
    param_const = setxor(1:length(q0),param);
    Xqnew(param_const,:) = q0(param_const,ones(1,2*m));
    
    %propagating Xqnew, the stencil with both parameters for estimation as
    %well as const. parameters
    Dtemp=zeros(L,2*m); 
    for j=1:2*m 
        Dtemp(:,j)=rk4(dynfun,dt,time(k-1),xstate(:,k-1),Xqnew(:,j)); 
    end
    
    % need to push Dtemp through the observation function  eq. 45
    D=zeros(size(data,2),2*m);
    for j=1:2*m
        D(:,j)=feval(obsfun,Dtemp(:,j));
    end
    
    % generate the prior estimate for the parameter, weighted mean. based
    % on the observation of the parameter through the dynamics. eq. 46
    dkhat=sum(D,2)/(2*m);
    
    % **** param measurement update equations ****
    %
    % this is where square root filter is different 
    % parameter estimate covariance update
    Lk = (1/sqrt(2*m))*(D - dkhat(:,ones(1,2*m)));   % eq. 53
    % covariance
    [unitar, Sdk] = qr([Lk Re]',0); % eq 52
    Sdk=Sdk';
    
    % cross covariance calculation; first step
    Xk = (1/sqrt(2*m))*(Xq - qminus(:,ones(1,2*m)));
    % cross covariance
    Pwd = Xk*Lk';
    
    % gain
    Wk = (Pwd/Sdk')/Sdk;
    
    % measurement update of parameters
    qplus=qminus+Wk*(data(k,:)'-dkhat); % eq. 50
    
    % measurement update of covariance
    qrterm = Xk - Wk*Lk;
    [foo, Sqplus] = qr([qrterm Wk*Re]',0);
    Sqplus = Sqplus';

    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    w(param,k) = qplus(:);
    w(param_const,k) = q0(param_const);
    
    % state correction
    xstate(:,k) = feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),w(:,k));

    %Lambda "forgetting factor" decay
    lambda = 0.996;
    variancefloor = 1e-9;
    Rk = max((1/lambda - 1)*Sqplus,variancefloor);
    
    % store filter and covariances
    qfilter(:,k)=qplus;
    Pq{k}=Sqplus;
    tsdq(k,:) = (sqrt(diag(Sqplus*Sqplus'))*3)';
end
    
out.qfilter=qfilter';
out.Pq=Pq;
out.time=time;
out.data=data;
out.tsd=tsdq;

function dx=rk4(rhs,h,t,x,q)

% fourth order explicit rk integrator

k1=feval(rhs,t,x,q);
k2=feval(rhs,t+h/2,x+h/2*k1,q);
k3=feval(rhs,t+h/2,x+h/2*k2,q);
k4=feval(rhs,t+h,x+h*k3,q);

dx=x+(h/6)*(k1+2*k2+2*k3+k4);    
    
function sigma = cubaturepoints_sr(m,P)    
    
n = length(m);
M = 2*n;
% constructs point set 'generator' 
ui = [eye(n), -eye(n)];

% constructs Xi
Xi = sqrt(M/2)*ui;


% construct cubature pts
sigma = zeros(n,M);
for i = 1:M
    sigma(:,i) = P*Xi(:,i) + m;
end
% end of cubaturepoints function