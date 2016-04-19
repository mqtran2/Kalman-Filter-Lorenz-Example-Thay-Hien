function out=dualckf(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0,param)
%
%DualUKF: Dual Estimation Cubature Kalman Filter
%     DUALCKF(dynfun,obsfun,data,time,x0,R,V,P0x,P0q,Re,Rr,q0)
%   This implements a continuous-discrete cubature kalman bucy filter to
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
    %time update of state and parameters
    xminus=xfilter(:,k-1);
    qminus=qfilter(:,k-1);
    %time update of covariances (x: state, q: parameter) 
    Pxminus=Px{k-1};
    Pqminus=Pq{k-1}+Rk;   % (7.79)    

    % Housekeeping: tracking which parameters are being estimated and which
    % are being held constant
    q(param,k-1) = qfilter(:,k-1);
    param_const = setxor(1:length(q0),param);
    q(param_const,k-1) = q0(param_const);

    
    % Cubature points for the parameters.  
    Xq=cubaturepoints(qminus,Pqminus);
    
    % Cubature points for the states 
	Xx=cubaturepoints(xminus,Pxminus);
    
    % now need to propagate the stencil into the dynamics using the
    % PRIOR estimate for the parameter (7.53 + qminus)
    
    Xhat=zeros(L,2*L);
    for j=1:2*L
        Xhat(:,j)=rk4(dynfun,dt,time(k-1),Xx(:,j),q(:,k-1)); % SigmaProp
    end
    
    % generate the prior estimate for the state, weighted mean. (7.54)
    xminus=sum(Xhat,2)/(2*L);
    
    % update the state covariance -- 7.55
    Pxminus = zeros(L,L);
    for i=1:2*L
        Xix = Xhat(:,i)*Xhat(:,i)';
        Pxminus = Pxminus + Xix;
    end
    Pxminus = Pxminus/(2*L) - xminus*xminus'+ V;
    
    % propogate the stencil for the parameters into the dynamics using the
    % PRIOR estimate for the states (7.81 + xminus).
    
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
    
    % need to push Dtemp through the observation function
    D=zeros(size(data,2),2*m);
    for j=1:2*m
        D(:,j)=feval(obsfun,Dtemp(:,j));
    end
    
    % generate the prior estimate for the parameter, weighted mean. based
    % on the observation of the parameter through the dynamics. 7.83
    dkhat=sum(D,2)/(2*m);
    
    % update the state stencil -- 7.56

    Xminus=cubaturepoints(xminus,Pxminus);

    % push the state stencil through the obsfun
    Y=zeros(size(data,2),2*L);  
    for j=1:2*L
        Y(:,j)=feval(obsfun,Xminus(:,j)); % 7.57
    end

    % prior estimate, yhat_minus, 7.58
	yminus=sum(Y,2)/(2*L); 
    
    % **** param measurement update equations ****
    
    % parameter estimate covariance update on d.
    % matrix version of 7.84
    Pdd=zeros(size(data,2),size(data,2));
    for i=1:2*m;
       Did = D(:,i)*D(:,i)';
       Pdd = Pdd + Did;
    end
    Pdd=Pdd/(2*m)-dkhat*dkhat'+ Re;
    
    
    % parameter cross variance update, comparing the original stencil with
    % the propagated stencil.

    Pwd=0;
    for i=1:2*m
        WD = Xq(:,i)*D(:,i)'/(2*m);
        Pwd = Pwd + WD;
    end
    Pwd = Pwd - qminus*dkhat';

    % compute the kalman gain for the parameters

	Wq=Pwd/Pdd; % 7.86
     
    % measurement update of parameter
    qplus=qminus+Wq*(data(k,:)'-dkhat); % (7.87)
    % measurement update of covariance of parameter
    Pqplus=Pqminus-Wq*Pdd*Wq'; % 7.88
    
	% **** state measurement update equations ****
    
    % predicted state innovation covariance matrix 
    Pzz=zeros(size(data,2),size(data,2));
    for i=1:2*L
        Ziz = Y(:,i)*Y(:,i)';
        Pzz = Pzz + Ziz;
    end
    Pzz = Pzz/(2*L) - yminus*yminus' + R;
    
    % predicted state cross-covariance matrix
    Pxz = 0;
    for i=1:2*L;
        XZ = Xminus(:,i)*Y(:,i)'/(2*L);
        Pxz = Pxz + XZ;
    end
    Pxz = Pxz - xminus*yminus';    
    % compute the gain for the states
    Wx=Pxz/Pzz; % 7.61
    
    % measurement update of covariance of states
    Pxplus=Pxminus-Wx*Pzz*Wx'; % 7.63
    % measurement update of filter state
    xplus=xminus+Wx*(data(k,:)'-yminus); % 7.62
    
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
    lambda = 0.996;
    variancefloor = 1e-9;
    Rk = max((1/lambda - 1)*Pqplus,variancefloor);
    
    %Anneal Method
    %annealfactor = 0.9;
    %variancefloor = 1e-9;
%    Rk = diag(max(annealfactor * diag(Rk) , variancefloor));
    
    
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