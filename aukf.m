function out=aukf(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Rq,Vq,q)
%
%AUKF: Adaptive Estimation Unscented Kalman Filter
%     AUKF(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0)
%   This implements a continuous-discrete unscented kalman bucy filter to
%   simultaneously estimate both the state of the system and the system
%   covariances.
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
%           Rq: Parameter Measurement noise covariance, constant
%           Vq: Parameter Process noise covariance, constant
%           q0: initial guess of parameter value
%           q: index of parameters to estimate
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

x0=x0(:); % columns, please. 
L=numel(x0);
m=numel(diag(V));
N=length(data);
dt=time(2)-time(1);

% init the filter
Px=cell(L,L,N);
Pq=cell(m,m,N);

% fill in the initial conditions for both filters.

xfilter(:,1)=x0;   % initial state filter value
qfilter(:,1)=diag(V);   % initial parameter filter value
Px{1}=P0x;  % initial state covariance
Pq{1}=P0q;  % initial parameter covariance
%Rk = Rr;    % initial Parameter process noise
tsdq(1,:) = (sqrt(diag(Pq{1}))*3)';
tsdx(1,:) = (sqrt(diag(Px{1}))*3)';

% main filter loop

for k=2:N;
    %time update of state and parameters
    xminus=xfilter(:,k-1);
    qminus=qfilter(:,k-1);
    %time update of covariances (x: state, q: diag of V) 
    Pxminus=Px{k-1};
    Pqminus=Pq{k-1};   % (7.79)    

    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    q(param,k-1) = qfilter(:,k-1);
    param_const = setxor([1:length(q0)],param);
    q(param_const,k-1) = q0(param_const);

    
    % Sigma points for the parameters.  (7.80)
    [Xq Wq Wmq]=sigmapoints(qminus,Pqminus,1,2,3-m);
    
    % Sigmapoints for the states (7.52)
	[Xx Wx Wmx]=sigmapoints(xminus,Pxminus,1,2,3-L);
    
    % now need to propagate the stencil into the dynamics using the
    % PRIOR estimate for the parameter (7.53 + qminus)
    
    Xhat=zeros(L,2*L+1);
    for j=1:2*L+1
        Xhat(:,j)=rk4(dynfun,dt,time(k-1),Xx(:,j),q); % SigmaProp
    end
    
    % generate the prior estimate for the state, weighted mean. (7.54)
    xminus=Xhat*Wmx;
    
    % update the state covariance -- 7.55
    Pxminus=Xhat*Wx*Xhat'+V;
    
    % propogate the stencil for the parameters into the dynamics using the
    % PRIOR estimate for the states (7.81 + xminus).
    
    % Changing stencil to include all parameters; Xqnew = sigma points for
    % parameters plus rows of the parameters which are held constant
    Xqnew = ones(length(q0),2*m+1);
    Xqnew(param,:) = Xq(:,:);
    Xqnew(param_const,:) = q0(param_const,ones(1,2*m+1));
    
    %propagating Xqnew, the stencil with both parameters for estimation as
    %well as const. parameters
    Dtemp=zeros(L,2*m+1); 
    for j=1:2*m+1 
        Dtemp(:,j)=rk4(dynfun,dt,time(k-1),xfilter(:,k-1),Xqnew(:,j)); 
    end
    
    % need to push Dtemp through the observation function
    D=zeros(size(data,2),2*m+1);
    for j=1:2*m+1
        D(:,j)=feval(obsfun,Dtemp(:,j));
    end
    
    % generate the prior estimate for the parameter, weighted mean. based
    % on the observation of the parameter through the dynamics. 7.83
    dkhat=D*Wmq;
    
    % update the state stencil -- 7.56

    Xminus=sigmapoints(xminus,Pxminus,1,2,3-L);

    % push the state stencil through the obsfun
    Y=zeros(size(data,2),2*L+1);  
    for j=1:2*L+1
        Y(:,j)=feval(obsfun,Xminus(:,j)); % 7.57
    end

    % prior estimate, yhat_minus, 7.58
	yminus=Y*Wmx; 
    
    % **** param measurement update equations ****
    
    % parameter estimate covariance update on d.
    % matrix version of 7.84

    Pdd=D*Wq*D'+Re;
    
    
    % parameter cross variance update, comparing the original stencil with
    % the propagated stencil.

    Pwd=Xq*Wq*D';
    
    % compute the kalman gain for the parameters

	Kq=Pwd/Pdd; % 7.86
     
    % measurement update of parameter
    qplus=qminus+Kq*(data(k,:)'-dkhat); % (7.87)
    % measurement update of covariance of parameter
    Pqplus=Pqminus-Kq*Pdd*Kq'; % 7.88
    
	% **** state measurement update equations ****
    
    Pyy=Y*Wx*Y'+R; % observation covariane (7.59)
    Pxy=Xminus*Wx*Y'; % observation cross variance % (7.60)
    
    % compute the gain for the states
    Kx=Pxy/Pyy; % 7.61
    
    % measurement update of covariance of states
    Pxplus=Pxminus-Kx*Pyy*Kx'; % 7.63
    % measurement update of filter state
    xplus=xminus+Kx*(data(k,:)'-yminus); % 7.62
    
%------------------------------------------------------------%
% 3 options for updating parameter process-noise covariance, Rk   %
%   1) robbins monro stochastic approximation
%         -fastest rate of absolute convergence, lowest MMSE
%   2) lambda-decay: Recursive least squares "forgetting factor"
%         -weights past data less, most appropriate on-line learning
%   3) aneal
%         -good final MMSE, but requires more monitoring and greater
%          prior knowledge of noise levels; good for on-line learning
%------------------------------------------------------------%
    
    %Robbins-Monro stochastic approximation scheme
    %for estimating innovation
%    Rk = (1-alpha)*Rk + alpha*K*(data(k,:)'-dkhat)*(data(k,:)'-dkhat)'*K';
    
    %Lambda "forgetting factor" decay
    %lambda = 0.996;
    %variancefloor = 1e-9;
    %Rk = max((1/lambda - 1)*Pqplus,variancefloor);
    
    %Anneal Method
    annealfactor = 0.9;
    variancefloor = 1e-9;
    Rk = diag(max(annealfactor * diag(Rk) , variancefloor));
    
    
    % store filters and covariances
    xfilter(:,k)=xplus;
    Px{k}=Pxplus;
    
    qfilter(:,k)=qplus;
    Pq{k}=Pqplus;
    
    tsdx(k,:)=(sqrt(diag(Px{k}))*3)';
    tsdq(k,:)=(sqrt(diag(Pq{k}))*3)';  
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

dx=x+h/6*(k1+2*k2+2*k3+k4);

function dx = eulerstep(rhs,h,t,x,q)

dx = x + h*feval(rhs,t,x,q);

function [X W wm wc sc] = sigmapoints(m, P, a, b, k, issqrt)

issqrt = 0;
if (nargin == 6)
    issqrt = 1;
end

if (nargin == 2)
    %a = 10^-3;
    a = 1;
    b = 2;
    k = 0;
    issqrt = 0;
end

n = length(m);
np = 2*n + 1;

c = a*a*(n+k);
sc = sqrt(c);

try
    P_sqrt = chol(P)';
catch ME
    P_sqrt=real(sqrtm(P));
    disp('flag');
end

%creation of sigma points
X = repmat(m, 1, np) + sc*[zeros(n,1) P_sqrt -P_sqrt];

% Form the weights
L = a*a*(n+k) - n;
wm = [L/(n+L) repmat(1/(2*(n+L)), 1, 2*n)]';
wc = [(L/(n+L) + (1 - a*a + b)) repmat(1/(2*(n+L)), 1, 2*n)]';

tmp = eye(np) - repmat(wm, 1, np);
W = tmp*diag(wc)*tmp';