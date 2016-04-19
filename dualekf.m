function out=dualekf(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0,param)
% 
%Dual EKBF: Dual Estimation Extended Kalman Filter
%
%   This implements a continuous-discrete extended kalman bucy filter 
% to estimate both the state of the system and the parameters of a nonlinear 
% observation function.   
%      INPUTS:
%           dynfun: rhs of ODE system (model) from which parameters are 
%                   being estimated (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           x0: initial condition for system
%           R:  Observation noise covariance, constant
%           V:  Process noise covariance, constant
%           P0x: initial state filter covariance
%           P0q: initial covariance of parameter values
%           Re: Measurement noise covariance, constant
%           Rr: Process noise covariance, constant
%           q0: initial guess of parameter value
%           param: index of parameters to estimate
%       OUTPUTS:
%           out.xfilter: state filter output
%           out.wfilter: filter of parameter estimates (last value is final
%                        estimate of parameter; best fit)
%           out.Px: State covariance Matrices for each time
%           out.Pw: covariance matrix at each filter point
%           out.time: time scale
%           out.data: original data used for estimation
%           out.tsdx: three standard deviations for state estimates
%           out.tsdq: three standard deviations for parameter estimates
%
% 
%
% implementation was done using 'Kalman Filtering and Neural Networks'
% edited by Haykin, chapter 5 (references in code refer to table 7.5)
%
% this code comes with no guarantee of any kind

if ischar(dynfun)==1; dynfun=str2func(dynfun); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end

x0=x0(:); q0=q0(:); % columns, please. 
L=numel(x0);
m=numel(q0(param));
N=length(data);
dt=time(2)-time(1);

% init the filter
Px=cell(L,L,N);
Pq=cell(m,m,N);

% fill in the initial conditions for both filters.

xfilter(:,1)=x0;   % initial state filter value
qfilter(:,1)=q0(param);   % initial parameter filter value
Px{1}=P0x;  % initial state covariance
Pq{1}=P0q;  % initial parameter covariance
Rk = Rr;    % initial Parameter process noise
tsdq(1,:) = (sqrt(diag(Pq{1}))*3)';
tsdx(1,:) = (sqrt(diag(Px{1}))*3)';

% initialize parameters (both those to be estimated and those held
% constant)
q = ones(length(q0),N);
q(:,1) = q0;
% main filter loop

for k=2:N;
    % time update of parameters
    qminus = qfilter(:,k-1);
    % time update of parameter covariance
    Pqminus = P{k-1}+Rk;
    
    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    q(param,k-1) = qfilter(:,k-1);
    param_const = setxor(1:length(q0),param);
    q(param_const,k-1) = q0(param_const);
    
    % Jacobian of nonlinear function for covariance update
    bigx=myAD(xfilter(:,k-1));
    bigF=feval(dynfun,time(k),bigx,q(:,k-1));
    A=getderivs(bigF);
    
    % time update of state 
    xminus = feval(@rk4,dynfun,dt,time(k),xfilter(:,k-1),q(:,k-1));
    % time update of state covariance
    Pxminus = A*P{k-1}*A' + V;
    
    %Jacobian of observation function
    xnew = myAD()
    
end
    