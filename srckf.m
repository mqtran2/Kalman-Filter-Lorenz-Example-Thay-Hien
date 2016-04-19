function out=srckf(dynfun,obsfun,data,time,x0,R,V,P0,q)

% SRCKF Square Root Cubature Kalman Bucy Filter.
%     CKF(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q)
%   This implements a continuous-discrete square root cubature kalman bucy filter to
%   estimate the state of the system
%      INPUTS:
%           dynfun: rhs of ODE system (model) including the parameters (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           x0: initial condition for system
%           R:  Observation noise covariance, constant
%           V:  Process noise covariance
%           P0: initial state filter covariance
%		    q: parameter values
%       OUTPUTS:
%           out.xfilter: state filter output 
%           out.P: State covariance matrices for each time
%           out.time: time scale
%           out.data: original data used for filter
%           out.tsd: +/- 3 std. deviations of state filter
%
% Brett Matzuka & Adam Attarian, NCSU, July 2009
%
% See 'Cubature Kalman Filters' by Simon Haykin
%
% This code comes with no guarantees of any kind

if ischar(dynfun)==1; dynfun=str2func(dynfun); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end

L=numel(x0); % this is the n from the comments above.
N=length(data);
dt=time(2)-time(1);
m = 2*L;

% lets init the filter. using a cell to store each filter covmat.
P=zeros(L,L,N);
xfilter=zeros(L,N);
tsd=zeros(N,L);

R = chol(R)';
V = chol(V)';

% fill in the initial conditions
xfilter(:,1)=x0;
P(:,:,1)=chol(P0)';
tsd(1,:)=(sqrt(diag(P(:,:,1)*P(:,:,1)')))';

for k=2:N;
    % time update of state
    xminus = xfilter(:,k-1);
    % time update of covariance
    Skminus = P(:,:,k-1);
    
    % create cubature points for prediction
    X=cubaturepoints_sr(xminus,Skminus);
    
    % propagate the cubature points through the model
    Xstar=zeros(L,m);
    for i=1:m
        Xstar(:,i) = feval(@rk4,dynfun,dt,time(k-1),X(:,i),q);
    end

    % predicted state
    xhat = sum(Xstar,2)/m;
    
    % this is where square root CKF differs
    %--------------------------------------
    Xstark = (1/sqrt(m))*(Xstar - xhat(:,ones(1,m)));
    [useless, SK_] = qr([Xstark V]',0);
    SK_ = SK_';
    %---------------------------------------
    
    % measurement update cubature points
    X2=cubaturepoints_sr(xhat,SK_);
    
    % propagate the cubature points through the observation function
    Z=zeros(size(data,2),m);
    for i=1:m;
        Z(:,i) = feval(obsfun,X2(:,i));
    end

    % predicted measurement
    zhat =sum(Z,2)/m;
    
    % this is where square root CKF differs
    %--------------------------------------
    LK = (1/sqrt(m))*(Z-zhat(:,ones(1,m)));
    % covariance update
    [useless2 Szz] = qr([LK R]',0);
    Szz = Szz';
    % cross covariance update
    XKK_ = (1/sqrt(m))*(X2 - xhat(:,ones(1,m)));
    Pxz = XKK_*LK';
    
    % gain
    Wk = (Pxz/Szz')/Szz;
    
    %---------------------------------------
    
    % measurement update of filter state
    xplus = xhat + Wk*(data(k,:)'-zhat);
    
    % square root update covariance
    %---------------------------------------
    [useless3 Skplus] = qr([XKK_ - Wk*LK Wk*R]',0);
    Skplus=Skplus';
    %---------------------------------------
    
    % store filter and covariances
    xfilter(:,k)=xplus;
    P(:,:,k)=Skplus;
    tsd(k,:) = (sqrt(diag(Skplus*Skplus')))';
    
end

out.xfilter=xfilter';
out.Px=P;
out.time=time;
out.data=data;
out.tsd=tsd;


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
