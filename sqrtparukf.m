function out=sqrtparukf(dynfun,obsfun,data,time,w0,Re,Rr,Pw0,xstate0,param)
% 
%Sqrt UKBF: Square Root Unscented Kalman Filter Parameter Estimation
%
%   This implements a continuous-discrete square root unscented kalman bucy filter 
% to estimate the parameters of a nonlinear observation function.  
% The states are defined to be the parameters of the system. 
%      INPUTS:
%           dynfun: rhs of ODE system (model) from which parameters are 
%                   being estimated (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for filter (column vector)
%           time: time period observations occur over
%           w0: initial guess of parameter value
%           Re: Measurement noise covariance, constant
%           Rr: Process noise covariance, constant
%           Pw0: initial covariance of parameter values
%           xstatex0: initial condition for model
%           param: index of parameters to estimate
%       OUTPUTS:
%           out.wfilter: filter of parameter estimates (last value is final
%                        estimate of parameter; best fit)
%           out.Pw: covariance matrix at each filter point
%           out.time: time scale
%           out.data: original data used for estimation
%           out.tsd: three standard deviations for estimates
%
% Brett Matzuka & Adam Attarian, NCSU, July 2009
%
% implementation was done using 'Kalman Filtering and Neural Networks'
% edited by Haykin, chapter 7 (references in code refer to table 7.5)
%
% this code comes with no guarantee of any kind

% initialize the filter.
w0=w0(:);
L = length(w0(param));
N=length(data);
dt=time(2)-time(1);
P=cell(L,L,N);
wfilter=zeros(L,N);
xstate = zeros(numel(xstate0),N);
% main filter loop.

%correct initial covariances
Rr = chol(Rr)';
Re = chol(Re)';

P{1}=chol(Pw0)';             % initial covariance value
wfilter(:,1)=w0(param);      % initial filter value
Rk = Rr;              % inital Process noise
xstate(:,1) = xstate0;
tsd(1,:) = (sqrt(diag(P{1}))*3)';

w = ones(length(w0),N);
w(:,1) = w0;

for k=2:N
    % annealing initial covariance
    %Dk = -diag(P{k-1}) + sqrt(diag(Rk) + diag(P{k-1}).^2);
    %Dk=diag(Dk);

    %time update of parameters
    wminus=wfilter(:,k-1); % 7.149 (pg. 276)
    %time update of covariance 
    lambda = 0.996;
    Pwminus=P{k-1}*lambda^(-1/2); % 7.150
    
    % generate the sigma points for the square root, eqn 7.151
    [X W Wm Wc] = sigmapoints_sqrt(wminus, Pwminus,1,2,3-L);
    
    % Changing stencil to include all parameters; Xqnew = sigma points for
    % parameters plus rows of the parameters which are held constant
    Xqnew = ones(length(w0),2*L+1);
    Xqnew(param,:) = X(:,:);
    param_const = setxor([1:length(w0)],param);
    Xqnew(param_const,:) = w0(param_const,ones(1,2*L+1));
    % propagate X, sigma points, through the RHS, 
    % and then eval with the obs fun.
    Dtemp=zeros(size(xstate,1),2*L+1);
    for j=1:2*L+1;
        Dtemp(:,j) = feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),Xqnew(:,j));
    end
    D = zeros(size(data,2),2*L+1);
    for i=1:2*L+1;
        D(:,i) = feval(obsfun,Dtemp(:,i));
    end
    
    % time update of parameter estimate wrt observation
    dkhat=D*Wm; % 7.153
    
    % measurement update of covariance
    %   this is where square root ukf differs from ukf
    %--------------------------------------------------
    D1 = D-dkhat(:,ones(1,2*L+1));
    sqrt_wc = sqrt(Wc(2));
    
    % first step in calculating obser covariance
    [unitar, Sdk] = qr([sqrt_wc*D1(:,2:2*L+1) Re]',0); % eq 7.154
    
%     Sdk = Sdk';
    if Wc(1)>0;
        Sdk = cholupdate(Sdk,sqrt(sqrt(abs(Wc(1))))*D1(:,1));
    else
        disp('downdate')
        Sdk = cholupdate(Sdk,sqrt(sqrt(abs(Wc(1))))*D1(:,1),'-');
    end
    Sdk = Sdk';
    
    % measurement update of crosscovariance
    Pwd=X*W*D';    % crossvariance 7.156
   
    % gain
    Kk = (Pwd/Sdk')/Sdk; % eqn 7.157
    
    % measurement update of parameter
    wplus = wminus + Kk*(data(k,:)'-dkhat);
    

    % U: correct covariance
    U = Kk*Sdk; % eqn 7.158
    
    Pwminus = Pwminus';
    % measurement update of covariance of parameter
    for i=1:size(data,2)
       Pwminus = cholupdate(Pwminus,U(:,i),'-');
    end
    Pwminus = Pwminus';
    Swplus = Pwminus;
    
    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    w(param,k) = wplus(:);
    w(param_const,k) = w0(param_const);
    
    % state correction
    xstate(:,k) = feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),w(:,k));

    %Lambda "forgetting factor" decay
    lambda = 0.996;
    variancefloor = 1e-9;
    Rk = max((1/lambda - 1)*Swplus,variancefloor);
    
    % store filter and covariances
    wfilter(:,k)=wplus;
    P{k}=Swplus;
    tsd(k,:) = (sqrt(diag(P{k}*P{k}'))*3)';
end

out.wfilter=wfilter';
out.Pw=P;
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
    