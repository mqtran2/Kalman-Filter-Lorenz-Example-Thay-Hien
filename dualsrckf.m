function out=dualsrckf(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0,param)
%
%DualSRUKF: Dual Estimation Square Root Cubature Kalman Filter
%     DUALSRCKF(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0)
%   This implements a continuous-discrete square root cubature kalman bucy filter to
%   simultaneously estimate both the state of the system and the system
%   parameteres.
%      INPUTS:
%           dynfun: rhs of ODE system (model) including the parameters (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           x0: initial condition for system
%           R:  Observation noise covariance, constant
%           V:  Process noise covariance
%           P0x: initial state filter covariance
%           P0q: initial parameter filter covariance
%           Re: Parameter Measurement noise covariance, constant
%           Rr: Parameter Process noise covariance, constant
%           w0: initial guess of parameter value
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
% See 'Cubature Kalman Filters' by Simon Haykin for more info
% on dual estimation and the CKF
%
% This code comes with no guarantees of any kind

if ischar(dynfun)==1; dynfun=str2func(dynfun); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end

x0=x0(:); q0=q0(:); % columns, please. 
L=numel(x0);
m=numel(q0(param));
N=length(data);
dt=time(2)-time(1);

% init the filter
Px=zeros(L,L,N);
Pq=zeros(m,m,N);

R = chol(R)';
V = chol(V)';
Re = chol(Re)';
Rr = chol(Rr)';
% fill in the initial conditions for both filters.

xfilter(:,1)=x0;   % initial state filter value
qfilter(:,1)=q0(param);   % initial parameter filter value
Px(:,:,1)=chol(P0x)';  % initial state covariance
Pq(:,:,1)=chol(P0q)';  % initial parameter covariance
Rk = Rr;    % initial Parameter process noise
tsdq(1,:) = (sqrt(diag(Pq(:,:,1)*Pq(:,:,1)')))';
tsdx(1,:) = (sqrt(diag(Px(:,:,1)*Px(:,:,1)')))';

% initialize parameters (both those to be estimated and those held
% constant)
q = ones(length(q0),N);
q(:,1) = q0;
% main filter loop

for k=2:N;
    %time update of state and parameters
    xminus=xfilter(:,k-1);
    qminus=qfilter(:,k-1);
     %time update of covariances (x: state, q: parameter) 
    Pxminus=Px(:,:,k-1);
    Pqminus=Pq(:,:,k-1)+Rk;
    
    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    q(param,k-1) = qfilter(:,k-1);
    param_const = setxor(1:length(q0),param);
    q(param_const,k-1) = q0(param_const);
    
    % Cubature points for the parameters.  
    Xq=cubaturepoints_sr(qminus,Pqminus);
    
    % Cubature points for the states 
	Xx=cubaturepoints_sr(xminus,Pxminus);
    
    % propagate the cubature points through the model
    Xstar=zeros(L,2*L);
    for i=1:2*L
        Xstar(:,i) = feval(@rk4,dynfun,dt,time(k-1),Xx(:,i),q(:,k-1));
    end

    % predicted state
    xhat = sum(Xstar,2)/(2*L);
    
    % this is where square root CKF differs
    % prediction covariance
    %--------------------------------------
    Xstark = (1/sqrt(2*L))*(Xstar - xhat(:,ones(1,2*L)));
    [useless, SK_] = qr([Xstark V]',0);
    SK_ = SK_';
    %---------------------------------------
    
    % Changing stencil to include all parameters; Xqnew = sigma points for
    % parameters plus rows of the parameters which are held constant
    Xqnew = ones(length(q0),2*m);
    Xqnew(param,:) = Xq(:,:);
    Xqnew(param_const,:) = q0(param_const,ones(1,2*m));
    
    
    %propagating Xqnew, the stencil with both parameters for estimation as
    %well as const. parameters
    Dtemp=zeros(L,2*m); 
    for j=1:2*m 
        Dtemp(:,j)=rk4(dynfun,dt,time(k-1),xfilter(:,k-1),Xqnew(:,j)); 
    end
    
    % need to push Dtemp through the observation function  eq. 45
    D=zeros(size(data,2),2*m);
    for j=1:2*m
        D(:,j)=feval(obsfun,Dtemp(:,j));
    end
    
    % generate the prior estimate for the parameter, weighted mean. based
    % on the observation of the parameter through the dynamics. eq. 46
    dkhat=sum(D,2)/(2*m);
    
    % measurement update cubature points for state filter
    X2=cubaturepoints_sr(xhat,SK_);
    
    % propagate the cubature points through the observation function
    Z=zeros(size(data,2),2*L);
    for i=1:2*L;
        Z(:,i) = feval(obsfun,X2(:,i));
    end

    % predicted measurement for state filter
    zhat =sum(Z,2)/(2*L);
    
    % **** param measurement update equations ****
    %
    % this is where square root filter is different 
    % parameter estimate covariance update
    %-------------------------------------------
    Lk = (1/sqrt(2*m))*(D - dkhat(:,ones(1,2*m)));   % eq. 53
    % covariance
    [unitar, Sdk] = qr([Lk Re]',0); % eq 52
    Sdk=Sdk';
    
    % cross covariance calculation; first step
    Xk = (1/sqrt(2*m))*(Xq - qminus(:,ones(1,2*m)));
    % cross covariance
    Pwd = Xk*Lk';
    
    % gain
    Wkq = (Pwd/Sdk')/Sdk;
    %-------------------------------------------
    
    % measurement update of parameters
    qplus=qminus+Wkq*(data(k,:)'-dkhat); % eq. 50
    
    % different from ckf
    % -------------------------------------------
    % measurement update of covariance
    qrterm = Xk - Wkq*Lk;
    [foo, Sqplus] = qr([qrterm Wkq*Re]',0);
    Sqplus = Sqplus';
    %-------------------------------------------
    
    % *****  State measurement Update equations ******
    
    % this is where square root CKF differs
    % state covariance update
    %--------------------------------------
    LK = (1/sqrt(2*L))*(Z-zhat(:,ones(1,2*L)));
    % covariance update
    [useless2 Szz] = qr([LK R]',0);
    Szz = Szz';
    % cross covariance update
    XKK_ = (1/sqrt(2*L))*(X2 - xhat(:,ones(1,2*L)));
    Pxz = XKK_*LK';
    
    % gain
    Wkx = (Pxz/Szz')/Szz;
    
    %---------------------------------------
    
    % measurement update of filter state
    xplus = xhat + Wkx*(data(k,:)'-zhat);
    
    % square root update covariance
    %---------------------------------------
    [useless3 Skplus] = qr([XKK_ - Wkx*LK Wkx*R]',0);
    Skplus=Skplus';
    %---------------------------------------
    
    %Lambda "forgetting factor" decay
    lambda = 0.996;
    variancefloor = 1e-9;
    Rk = max((1/lambda - 1)*Sqplus,variancefloor);
       
    % store filters and covariances
    xfilter(:,k)=xplus;
    Px(:,:,k)=Skplus;
    
    qfilter(:,k)=qplus;
    Pq(:,:,k)=Sqplus;
    
    tsdx(k,:)=(sqrt(diag(Px(:,:,k)*Px(:,:,k)')))';
    tsdq(k,:)=(sqrt(diag(Pq(:,:,k)*Pq(:,:,k)')))';  
    
end

out.xfilter=xfilter';
out.qfilter=qfilter';
out.Px=Px;
out.Pq=Pq;
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
