function out = parPF(dynfun,obsfun,data,time,w0,Rmod,Rmea,xstate0,N,param)

%      INPUTS:
%           dynfun: rhs of ODE system (model) from which parameters are 
%                   being estimated (column vector)
%           obsfun: general function for observations (d = G(x,w))
%           data: data points used for particle filter (column vector)
%           time: time period observations occur over
%           w0: initial guess of parameter value
%           Rmod: parameter noise covariance, constant
%           Rmea: measurement noise covariance, constant
%           xstatex0: initial condition for model
%           N: # of particles
%           param: index of parameters to estimate
%       OUTPUTS:
%           out.filter: filter of parameter estimates (last value is final
%                        estimate of parameter; best fit)
%           out.stand_dev: standard deviation at each filter point
%           out.time: time scale
%           out.data: original data used for estimation



% initialize filter 
w0 = w0(:);             % initial state of parameter
dt = time(2)-time(1);   % time step
D = length(data);       % # of iterations
L = length(w0(param));  % number of params to estimate
filter = ones(L,D);    % initialize filter
stand_dev = ones(L,D);  % initialize stand_dev
xstate = zeros(numel(xstate0),D); % model values
xstate(:,1) = xstate0;  % initialize model

% preallocated matrices and values
R = inv(Rmea);
R1 = inv(chol(det(Rmea)));
tau = 1; 

% parameters not to be estimated
param_const = setxor(1:length(w0),param); 
% setting up prior distribution for parameters to be estimated
prior = ones(length(w0),N);    
prior(param,:) = repmat(w0(param,:),1,N)+Rmod*randn(L,N);
prior(param_const,:) = w0(param_const,ones(1,N));

% initial value of filter and deviation
filter(:,1) = w0(param);    
stand_dev(:,1) = sqrt(diag(Rmod));

%preallocate the weights, qi
qi = zeros(2,N);

%main loop
for k = 2:D
    % compute the likelihood of particles and weights
    % SIS: step 1
    for i = 1:N
        % propagating particles through model
        model(:,i) = feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),prior(:,i));
        % propagating model particles through observation
        obser(:,i) = feval(obsfun,model(:,i));
        % likelihood function (importance weights): prob. density
        qi(2,i) = (R1/(2*pi)^(L/2))*exp(-0.5*(data(k,:)'-obser(:,i))'*R*(data(k,:)'-obser(:,i)));
    end
    qi_sum = sum(qi(2,:));
    qi(2,:) = qi(2,:)/qi_sum;
    
    % importance resampling for posterier
    % (not used for stationary processes s.a. PE)
%     for i = 1:N
%         U = rand;
%     end
%     
    % calculating KL-distance: Step 2
    kappa  = qi(2,:)'*(qi(2,:)-qi(1,:));
    
    % setting up KL distance calculation for following iterations
    if k == 2
        kappa = 0;
        qi(1,:) = qi(2,:);
    end
    % if KL-distance is not met, run MCMC for posterier
    % Step 3
    if kappa > tau
        mu = prior*qi(2,:)';
        sigma = qi(2,:)'*(prior-mu)*(prior-mu)';  % NOT RIGHT
        posterior = MCMC();   % NEEDS TO BE WRITTEN
        prior(param,:) = posterior;
        prior(param_const,:) = w0(param_const,ones(1,N));
    end
    
    %state correction
    xstate(:,k) = feval(@rk4,dynfun,dt,time(k-1),xstate(:,k-1),mean(prior,2));
    
    %store filter and standard deviation
    filter(:,k) = mean(prior(param,:),2);
    stand_dev(:,k) = std(prior(param,:),0,2);
    
end

out.filter = filter';
out.stand_dev = stand_dev;
out.time = time;
out.data = data;

function dx=rk4(rhs,h,t,x,q)

% fourth order explicit rk integrator

k1=feval(rhs,t,x,q);
k2=feval(rhs,t+h/2,x+h/2*k1,q);
k3=feval(rhs,t+h/2,x+h/2*k2,q);
k4=feval(rhs,t+h,x+h*k3,q);

dx=x+(h/6)*(k1+2*k2+2*k3+k4);

function posterior = MCMC()