function out=srukf(dynfun,obsfun,data,time,x0,R,V,Px0,q)
% 
%Sqrt UKBF: Square Root Unscented Kalman Filter
%
%   This implements a continuous-discrete square root unscented kalman bucy filter 
% to estimate the state of a nonlinear observation function.  
%  
%      INPUTS:
%           dynfun: rhs of ODE system (model) from which parameters are 
%                   being estimated (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           x0: initial guess of parameter value
%           R: Measurement noise covariance, constant
%           V: Process noise covariance, constant
%           Px0: initial covariance of parameter values
%           q: parameter values
%       OUTPUTS:
%           out.xfilter: filter of parameter estimates (last value is final
%                        estimate of parameter; best fit)
%           out.Px: covariance matrix at each filter point
%           out.time: time scale
%           out.data: original data used for estimation
%           out.tsd: three standard deviations for estimates
%
% Brett Matzuka & Adam Attarian, NCSU, September 2009
%
% implementation was done using 'Kalman Filtering and Neural Networks'
% edited by Haykin, chapter 7
%
% this code comes with no guarantee of any kind

% initialize the filter.
x0 = x0(:);
L = length(x0);
N=length(data);
dt=time(2)-time(1);
Px=zeros(L,L,N);
xfilter=zeros(L,N);

% main filter loop.

%correct initial covariances
R = chol(R)';
V = chol(V)';

Px(:,:,1)=chol(Px0)';             % initial covariance value
xfilter(:,1)=x0;      % initial filter value
tsd(1,:) = (sqrt(diag(Px(:,:,1))))';

for k=2:N
    %time update of state
    xminus=xfilter(:,k-1);
    %time update of state covariance
    Pxminus=Px(:,:,k-1);
    
    %making sigma points for model
    [X W Wm Wc] = sigmapoints_sqrt(xminus, Pxminus,1,2,3-L);
    
    %passing sigma point stencil through the dynamic model and propagating
    %forward
    Xhat=zeros(L,2*L+1);
    for j=1:2*L+1        
        Xhat(:,j)=rk4(dynfun,dt,time(k-1),X(:,j),q); % SigmaProp
        %[tless, Xless] = ode15s(dynfun,[time(k-1) time(k)],X(:,j),[],q);
        %Xhat(:,j) = Xless(end,:);
    end
    
    % generate the prior estimate for the state, weighted mean. (7.54)
    xhat_=Xhat*Wm;
    
    % measurement update of covariance
    %   this is where square root ukf differs from ukf
    %--------------------------------------------------
    Xstar = Xhat-xhat_(:,ones(1,2*L+1));
    sqrt_wc = sqrt(Wc(2));

    % qr factorization
    [waste, Sk_] = qr([sqrt_wc*Xstar(:,2:2*L+1) V]',0); % eq 7.154
    
    %cholesky update to complete the covariance prediction
    if Wc(1)>0;
        Sk_ = cholupdate(Sk_,sqrt(sqrt(abs(Wc(1))))*Xstar(:,1));
    else
        disp('downdate')
        Sk_ = cholupdate(Sk_,sqrt(sqrt(abs(Wc(1))))*Xstar(:,1),'-');
    end
    Sk_ = Sk_';  % requires lower triangular form
    
    %making sigma points for observations
    Xminus = sigmapoints_sqrt(xhat_, Sk_,1,2,3-L);
    
    % push the state stencil through the obsfun
    Y=zeros(size(data,2),2*L+1);  
    for j=1:2*L+1
        Y(:,j)=feval(obsfun,Xminus(:,j)); % 7.57
    end
    
    % prior estimate for observation
	yhat_=Y*Wm;
    
    % measurement update of covariance
    %   this is where square root ukf differs from ukf
    %--------------------------------------------------
    Ystar = Y-yhat_(:,ones(1,2*L+1));

    % qr factorization 
    [waste, Syk] = qr([sqrt_wc*Ystar(:,2:2*L+1) R]',0);

    %cholesky update to complete the observation: Observation Covariance
    if Wc(1)>0;
        Syk = cholupdate(Syk,sqrt(sqrt(abs(Wc(1))))*Ystar(:,1));
    else
        disp('downdate')
        Syk = cholupdate(Syk,sqrt(sqrt(abs(Wc(1))))*Ystar(:,1),'-');
    end
    Syk=Syk';
    
    % Observation Cross Variance
    Pxy = Xminus*W*Y';
    
    %Gain matrix
    Kk = (Pxy/Syk')/Syk;
    
    %measurement update of state filter
    xplus=xhat_+Kk*(data(k,:)'-yhat_);
    
    %U: correct Covariance
    U = Kk*Syk;
    
    % measurement update of covariance of parameter
    Sk_ = Sk_';
    for i=1:size(data,2)
       Sk_ = cholupdate(Sk_,U(:,i),'-');
    end
    Swplus = Sk_';
    
    % store filter and covariances
    xfilter(:,k)=xplus;
    Px(:,:,k)=Swplus;
    tsd(k,:) = (sqrt(diag(Swplus*Swplus')))';
end
    
  
out.xfilter=xfilter';
out.Px=Px;
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

    