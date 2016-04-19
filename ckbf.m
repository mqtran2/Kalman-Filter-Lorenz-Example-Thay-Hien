function out=ckbf(dynfun,obsfun,data,time,x0,R,V,P0,q)

% CKBF Generalized Cubature Kalman Bucy Filter.
%     CKF(dynfun,obsfun,data,time,x0,R,V,P0x,q)
%   This implements a continuous-discrete cubature kalman bucy filter to
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
% 
%
% See 'Cubature Kalman Filters' by Simon Haykin
%
% This code comes with no guarantees of any kind

if ischar(dynfun)==1; rhs=str2func(dynfun); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end

L=numel(x0); % this is the n from the comments above.
N=length(data);
dt=time(2)-time(1);
m = 2*L;

% lets init the filter. using a cell to store each filter covmat.
P=cell(L,L,N);
xfilter=zeros(L,N);
tsd=zeros(N,L);

% fill in the initial conditions
xfilter(:,1)=x0;
P{1}=P0;
tsd(1,:)=(sqrt(diag(P{1}))*3)';

for k=2:N;
    % time update of state
    xminus = xfilter(:,k-1);
    % time update of covariance
    Pkminus = P{k-1};

    % construct cubature points
    X=cubaturepoints(xminus,Pkminus);
    
    % propagate the cubature points through the model
    Xstar=zeros(L,m);
    for i=1:m
        Xstar(:,i) = feval(@rk4,dynfun,dt,time(k-1),X(:,i),q);
    end

    % predicted state
    xhat = sum(Xstar,2)/m;
    % predicted covariance
    Pk = zeros(L,L);
    for i=1:m
        Xix = Xstar(:,i)*Xstar(:,i)';
        Pk = Pk + Xix;
    end
    Pk = Pk/m - xhat*xhat'+ V;

    % measurement update
    X2=cubaturepoints(xhat,Pk);

    % propagate the cubature points through the observation
    Z=zeros(size(data,2),m);
    for i=1:m;
        Z(:,i) = feval(obsfun,X2(:,i));
    end

    % predicted measurement
    zhat =sum(Z,2)/m;
    % predicted innovation covariance matrix 
    Pzz=zeros(size(data,2),size(data,2));
    for i=1:m
        Ziz = Z(:,i)*Z(:,i)';
        Pzz = Pzz + Ziz;
    end
    Pzz = Pzz/m - zhat*zhat' + R;
    
    % predicted cross-covariance matrix
    Pxz = 0;
    for i=1:m;
        XZ = X2(:,i)*Z(:,i)'/m;
        Pxz = Pxz + XZ;
    end
    Pxz = Pxz - xhat*zhat';
    
    % Kalman Gain
    Wk = Pxz/Pzz;
    
    % measurement update of filter state
    xplus = xhat + Wk*(data(k,:)'-zhat);
    % measurement update of covariance of state
    Pplus = Pk - Wk*Pzz*Wk';
    
    % store fiters and covariance
    xfilter(:,k) = xplus;
    P{k} = Pplus;
    tsd(k,:) = (sqrt(diag(P{k}*P{k}'))*3);   
end

out.xfilter=xfilter';
out.P=P;
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


function sigma = cubaturepoints(m,P)

% calculates matrix square root
if any(eig(P)<=0) || ~isequal(P,P')
   S=real(sqrtm(P));
   disp('flag');
else
    try
        S = chol(P)';
    catch ME
        keyboard
    end
end

n = length(m);
M = 2*n;
% constructs point set 'generator' 
ui = [eye(n), -eye(n)];

% constructs Xi
Xi = sqrt(M/2)*ui;


% construct cubature pts
sigma = zeros(n,M);
for i = 1:M
    sigma(:,i) = S*Xi(:,i) + m;
end
% end of cubaturepoints function