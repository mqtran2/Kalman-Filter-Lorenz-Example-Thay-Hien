function out=parukf(dynfun,obsfun,data,time,w0,Re,Rr,Pw0,xstate0,param)
% 
%UKBF: Unscented Kalman Filter Parameter Estimation
%
%   This implements a continuous-discrete unscented kalman bucy filter 
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

P{1}=Pw0;             % initial covariance value
wfilter(:,1)=w0(param);      % initial filter value
Rk = Rr;              % inital Process noise
xstate(:,1) = xstate0;
tsd(1,:) = (sqrt(diag(P{1}))*3)';

w = ones(length(w0),N);
w(:,1) = w0;

for k=2:N
    
    %time update of parameters
    wminus=wfilter(:,k-1); % 7.78
    %time update of covariance 
    Pwminus=P{k-1}+Rk; % p 242, 7.79
    
    % generate sigma points, eqn 7.80
    [X W Wm] =sigmapoints(wminus,Pwminus,1,2,3-L); 

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
        [tless, Dless] = ode15s(dynfun,[time(k-1) time(k)],xstate(:,k-1),[],Xqnew(:,j));
        Dtemp(:,j) = Dless(end,:);
        %Dtemp(:,j)=feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),Xqnew(:,j)); 
    end
    
    for i = 1:2*L+1
        D(:,i) = feval(obsfun,Dtemp(:,i));
    end
    
    % time update of parameter estimate wrt obser
    dkhat=D*Wm; % 7.82
    
    % measurement update of covariance 
    Pdd=D*W*D'+ Re;   % covariance 7.84
   
    % measurement update of crosscovariance
    Pwd=X*W*D';    % crossvariance 7.85

    % gain
    K=Pwd/Pdd;
   
    % measurement update of parameter
    wplus=wminus+K*(data(k,:)'-dkhat);

    % measurement update of covariance of parameter
    Pwplus=Pwminus-K*Pdd*K';
    
    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    w(param,k) = wplus(:);
    w(param_const,k) = w0(param_const);
    
    % state correction
    xstate(:,k) = feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),w(:,k));
    
%-------------------------------------------------------%
% 3 options for updating process-noise covariance, Rk   %
%   1) robbins monro stochastic approximation
%         -fastest rate of absolute convergence, lowest MMSE
%   2) lambda-decay: Recursive least squares "forgetting factor"
%         -weights past data less, most appropriate on-line learning
%   3) aneal
%         -good final MMSE, but requires more monitoring and greater
%          prior knowledge of noise levels; good for on-line learning
%--------------------------------------------------------%
    
    %Robbins-Monro stochastic approximation scheme
    %for estimating innovation
%    Rk = (1-alpha)*Rk + alpha*K*(data(k,:)'-dkhat)*(data(k,:)'-dkhat)'*K';
    
    %Lambda "forgetting factor" decay
    lambda = 0.996;
    variancefloor = 1e-9;
    Rk = max((1/lambda - 1)*Pwplus,variancefloor);
    
    %Anneal Method
    %annealfactor = 0.9;
    %variancefloor = 1e-9;
%    Rk = diag(max(annealfactor * diag(Rk) , variancefloor));
    
    % store filter and covariances
    wfilter(:,k)=wplus;
    P{k}=Pwplus;
    tsd(k,:) = (sqrt(diag(P{k}))*3)';
end

out.wfilter=wfilter';
out.Pw=P;
out.time=time;
out.data=data;
out.tsd = tsd;


function dx=rk4(rhs,h,t,x,q)

% fourth order explicit rk integrator

k1=feval(rhs,t,x,q);
k2=feval(rhs,t+h/2,x+h/2*k1,q);
k3=feval(rhs,t+h/2,x+h/2*k2,q);
k4=feval(rhs,t+h,x+h*k3,q);

dx=x+(h/6)*(k1+2*k2+2*k3+k4);

function [X W wm wc sc] = sigmapoints(m, P, a, b, k, issqrt)

if (nargin == 2)
    a = 1;
    b = 3;
    k = 0;
end

n = length(m);
np = 2*n + 1;

c = a*a*(n+k);
sc = sqrt(c);

if any(eig(P)<=0) || ~isequal(P,P')
   P_sqrt=real(sqrtm(P));
   disp('flag');
else
    try
        P_sqrt = chol(P)';
    catch ME
        keyboard
    end
end

X = repmat(m, 1, np) + sc*[zeros(n,1) P_sqrt -P_sqrt];

% Form the weights
L = a*a*(n+k) - n;
wm = [L/(n+L) repmat(1/(2*(n+L)), 1, 2*n)]';
wc = [(L/(n+L) + (1 - a*a + b)) repmat(1/(2*(n+L)), 1, 2*n)]';

tmp = eye(np) - repmat(wm, 1, np);
W = tmp*diag(wc)*tmp';

function [X W wm wc sc] = sigmapts(m, P, V, N, a, b, k, issqrt)

if (nargin == 4)
    a = 1;
    b = 3;
    k = 0;
end

L = length(m) + length(V) + length(N);
np = 2*L+1;

den = a*a*(L+k);
L_lambda_sqrt = sqrt(den);

if any(eig(P)<=0) || ~isequal(P,P')
   P_sqrt=real(sqrtm(P));
   disp('flag');
else
    try
        P_sqrt = chol(P)';
    catch ME
        keyboard
    end
end

Sv = chol(V);   
Sn = chol(N);

Vmu = zeros(length(V),1);
Nmu = zeros(length(N),1);

X = cvecrep([m;Vmu;Nmu],np);

P_av = [P_sqrt zeros(length(Sv)); zeros(length(Sv)) Sv];
Pa = [P_av zeros(length(P_sqrt)+length(Sv),length(Sn)); zeros(length(Sn),length(Sv)+length(P_sqrt)) Sn];
Pa_sqrt = Pa*L_lambda_sqrt;
Pa_all = [Pa_sqrt -Pa_sqrt];


X(:,2:end) = X(:,2:end) + Pa_all;

lambda = den-L;
wm = [lambda/(L+lambda) repmat(1/(2*(L+lambda)), 1, 2*L)]';
wc = [(lambda/(L+lambda) + (1 - a*a + b)) repmat(1/(2*(L+lambda)), 1, 2*L)]';

tmp = eye(np) - repmat(wm, 1, np);
W = tmp*diag(wc)*tmp';

function dx = eulerstep(rhs,h,t,x,q)

dx = x + h*feval(rhs,t,x,q);
