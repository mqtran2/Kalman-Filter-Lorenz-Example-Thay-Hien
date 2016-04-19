function out=dualsrukf(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0,param)
%
%DualSRUKF: Dual Estimation Square Root Unscented Kalman Filter
%     DUALSRUKF(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0)
%   This implements a continuous-discrete square root unscented kalman bucy filter to
%   simultaneously estimate both the state of the system and the system
%   parameteres.  Square root implementation is computationally more stable
%   and garan
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
% See 'Kalman Filtering and Neural Networks' edited by Haykin for more info
% on dual estimation and the UKF
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

% correct initial covariances
Rr = chol(Rr)';
Re = chol(Re)';
R = chol(R)';
V = chol(V)';

% fill in the initial conditions for both filters.

xfilter(:,1)=x0;   % initial state filter value
qfilter(:,1)=q0(param);   % initial parameter filter value
Px(:,:,1)=chol(P0x)';  % initial state covariance
Pq(:,:,1)=chol(P0q)';  % initial parameter covariance
tsdq(1,:) = (sqrt(diag(Pq(:,:,1)*Pq(:,:,1)')))';
tsdx(1,:) = (sqrt(diag(Px(:,:,1)*Px(:,:,1)')))';

% initialize parameters (both those to be estimated and those held
% constant)
q = ones(length(q0),N);
q(:,1) = q0;
% main filter loop
options = odeset('RelTol',1e-6,'AbsTol',1e-6);
for k=2:N;
    %time update of state and parameters
    xminus=xfilter(:,k-1);
    qminus=qfilter(:,k-1);
    %time update of covariances (x: state, q: parameter) 
    Pxminus=Px(:,:,k-1);  
    % parameter covariance
    lambda = 0.996;
    Pqminus=Pq(:,:,k-1)*lambda^(-1/2);

    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    q(param,k-1) = qfilter(:,k-1);
    param_const = setxor(1:length(q0),param);
    q(param_const,k-1) = q0(param_const);
    
    %making sigma points for model
    [Xx Wx Wmx Wcx] = sigmapoints_sqrt(xminus, Pxminus,1,2,3-L);
    
    %making sigma points for parameters
    [Xq Wq Wmq Wcq] = sigmapoints_sqrt(qminus, Pqminus,1,2,3-L);
    
    %passing sigma point stencil through the dynamic model and propagating
    %forward using previous parameter filter values
    Xhat=zeros(L,2*L+1);
    for j=1:2*L+1
        Xhat(:,j)=rk4(dynfun,dt,time(k-1),Xx(:,j),q(:,k-1)); % SigmaProp
        %[tless, Xsol] = ode15s(dynfun,[time(k-1) time(k)],Xx(:,j),options,q(:,k-1));
        %Xhat(:,j) = Xsol(end,:);
    end
    
    % generate the prior estimate for the state, weighted mean. (7.54)
    xhat_=Xhat*Wmx;
    
    
    % measurement update of covariance for states
    %   this is where square root ukf differs from ukf
    %--------------------------------------------------
    Xstar = Xhat-xhat_(:,ones(1,2*L+1));
    sqrt_wcx = sqrt(Wcx(2));

    % qr factorization for state filter
    [waste, Sk_] = qr([sqrt_wcx*Xstar(:,2:2*L+1) V]',0); % eq 7.154
    
    %cholesky update to complete the state filter covariance prediction
    if Wcx(1)>0;
        Sk_ = cholupdate(Sk_,sqrt(sqrt(abs(Wcx(1))))*Xstar(:,1));
    else
        disp('downdate')
        Sk_ = cholupdate(Sk_,sqrt(sqrt(abs(Wcx(1))))*Xstar(:,1),'-');
    end
    Sk_ = Sk_';  % requires lower triangular form
    
    % Changing stencil to include all parameters; Xqnew = sigma points for
    % parameters plus rows of the parameters which are held constant
    Xqnew = ones(length(q0),2*m+1);
    Xqnew(param,:) = Xq(:,:);
    Xqnew(param_const,:) = q0(param_const,ones(1,2*m+1));
    
    % propagate Xqnew, sigma points, through the RHS, 
    % and then eval with the obs fun.
    Dtemp=zeros(L,2*m+1);
    for j=1:2*m+1;
        Dtemp(:,j) = feval(@rk4,dynfun,dt,time(k-1),xfilter(:,k-1),Xqnew(:,j));
        %[tless, Dsol] = ode15s(dynfun,[time(k-1) time(k)],xfilter(:,k-1),options,Xqnew(:,j));
        %Dtemp(:,j) = Dsol(end,:);
    end
    
    D=zeros(size(data,2),2*m+1);
    for i=1:2*m+1;
        D(:,i) = feval(obsfun,Dtemp(:,i));
    end
    
    % time update of parameter estimate wrt observation
    dkhat=D*Wmq; 
    
    %making sigma points for state filter observations
    Xminus = sigmapoints_sqrt(xhat_, Sk_,1,2,3-L);
    
    % push the state filter stencil through the obsfun
    Y=zeros(size(data,2),2*L+1);  
    for j=1:2*L+1
        Y(:,j)=feval(obsfun,Xminus(:,j)); % 7.57
    end
    
    % prior estimate for observation
	yhat_=Y*Wmx;
    
    % **** param measurement update equations ****
    
    % parameter estimate covariance update 
    % measurement update of covariance
    %   this is where square root ukf differs from ukf
    %--------------------------------------------------
    D1 = D-dkhat(:,ones(1,2*m+1));
    sqrt_wcq = sqrt(Wcq(2));
    
    % first step in calculating obser covariance for parameters
    [unitar, Sdk] = qr([sqrt_wcq*D1(:,2:2*m+1) Re]',0); % eq 7.154
    
    if Wcq(1)>0;
        Sdk = cholupdate(Sdk,sqrt(sqrt(abs(Wcq(1))))*D1(:,1));
    else
        disp('downdate')
        Sdk = cholupdate(Sdk,sqrt(sqrt(abs(Wcq(1))))*D1(:,1),'-');
    end
    Sdk = Sdk';
    
    % measurement update of crosscovariance
    Pwd=Xq*Wq*D';    % crossvariance
   
    % gain
    Kkq = (Pwd/Sdk')/Sdk; 
    
    % measurement update of parameter
    qplus = qminus + Kkq*(data(k,:)'-dkhat);
    

    % U: correct covariance
    Uq = Kkq*Sdk; % eqn 7.158
    
    Pqminus = Pqminus';
    % measurement update of parameter filter covariance
    for i=1:size(data,2)
       Pqminus = cholupdate(Pqminus,Uq(:,i),'-');
    end
    Sqplus = Pqminus';
    
    % **** state measurement update equations ****
    
    % measurement update of state filter covariance
    %   this is where square root ukf differs from ukf
    %--------------------------------------------------
    Ystar = Y-yhat_(:,ones(1,2*L+1));

    % qr factorization 
    [waste, Syk] = qr([sqrt_wcx*Ystar(:,2:2*L+1) R]',0);

    %cholesky update to complete the observation: Observation Covariance
    if Wcx(1)>0;
        Syk = cholupdate(Syk,sqrt(sqrt(abs(Wcx(1))))*Ystar(:,1));
    else
        disp('downdate')
        Syk = cholupdate(Syk,sqrt(sqrt(abs(Wcx(1))))*Ystar(:,1),'-');
    end
    Syk=Syk';
    
    % Observation Cross Variance
    Pxy = Xminus*Wx*Y';
    
    %Gain matrix
    Kkx = (Pxy/Syk')/Syk;
    
    %measurement update of state filter
    xplus=xhat_+Kkx*(data(k,:)'-yhat_);
    
    %U: correct Covariance
    Ux = Kkx*Syk;
    
    % measurement update of covariance of parameter
    Sk_ = Sk_';
    for i=1:size(data,2)
       Sk_ = cholupdate(Sk_,Ux(:,i),'-');
    end
    Swplus = Sk_';
    
    % store state and parameter filter and covariances
    xfilter(:,k)=xplus;
    Px(:,:,k)=Swplus;
    tsdx(k,:) = (sqrt(diag(Px(:,:,k)*Px(:,:,k)')))';
    
    qfilter(:,k)=qplus;
    Pq(:,:,k)=Sqplus;
    tsdq(k,:) = (sqrt(diag(Pq(:,:,k)*Pq(:,:,k)')))';
    
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


function [X W wm wc sc] = sigmapoints_sqrt(m, P, a, b, k, issqrt)

if (nargin == 2)
    a = 1e-3;
    b = 2;
    k = 0;
end

n = length(m);
np = 2*n + 1;

c = a*a*(n+k);
sc = sqrt(c);


X = repmat(m, 1, np) + sc*[zeros(n,1) P -P];

% Form the weights
L = a*a*(n+k) - n;
wm = [L/(n+L) repmat(1/(2*(n+L)), 1, 2*n)]';
wc = [(L/(n+L) + (1 - a*a + b)) repmat(1/(2*(n+L)), 1, 2*n)]';

tmp = eye(np) - repmat(wm, 1, np);
W = tmp*diag(wc)*tmp';
  