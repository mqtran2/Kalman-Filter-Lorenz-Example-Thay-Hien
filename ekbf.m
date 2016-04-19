function out = ekbf(model,obser,data,time,R,Q,x0,P0,q,fdad,options)

% Brett Matzuka, NCSU, May 2009
%
% Thanks to Adam Attarian for help in the update
% Thanks to Hien Tran for overseeing and his patience
%
%EKBF Extended Kalman Bucy Filter
%
%  this implements a continuous-discrete extended kalman filter 
%  using Automatic Differentiation to calculate the jacobian
%
%     INPUTS:
%          model: rhs of ODE system (include values for parameters)
%          obser: general function for observations (z = h(x))
%          data: data points used for filter (column vectors)
%          time: time period observations occur over
%          R: covariance noise for data, constant
%          Q: covariance noise for process, constant
%          x0: initial condition for model
%          P0: initial condition for covariance
%          q: parameter values
%          fdad: describes how to calculate jacobian
%               fdad = 0 (default); finite difference
%               fdad = 1; Automatic differentiation jacobian
%          options: ode solver options
%     OUTPUTS:
%          out.xfilter: filter output
%          out.time: time scale
%          out.P: covariance matrices for each time
%          out.sd: +/- standard deviations
fdad = fdad>0;
if nargin<10
    fdad = 0;
end
if nargin<11
    options = odeset('AbsTol',1e-4,'RelTol',1e-5);
end

%initialize the variables
N = length(data);
L = numel(x0);
xc = zeros(L,N);
pc = zeros(L,L,N);
sd = zeros(N,L);

%assign the initial conditions
xc(:,1) = x0;
pc(:,:,1) = P0;
sd(1,:) = (sqrt(diag(pc(:,:,1))))';

% run extended filter with finite difference jacobian
if fdad == 0;
    for i = 2:N;
        %PREDICTION:
        %predicton of state dynamics

        [tp, xp] = ode45(model,[time(i-1) time(i)],xc(:,i-1)',options,q);

        %initial condition for the covariance prediction
        initP = reshape(pc(:,:,i-1),1,L^2);

        %covariance differential equation predictor
        [tp, pp] = ode15s(@cov_fd,[time(i-1) time(i)],initP,options,model,xc,L,Q,q);
        pmat = reshape(pp(end,:),L,L);

        %jacobian of the observation function
        xnew = xp(end,:);
        Cjac = diffjac(obser,xnew,time(i),q);
        hobs = feval(obser,time(i),xnew,q);

        %CORRECTION:
        %kalman gain
        %[sm, lam, xn] = svd(Cjac*pmat*Cjac'+R);
        %invmat = xn*inv(lam)*sm';
        %K = pmat*Cjac'*invmat;
        %LL=chol(Cjac*pmat*Cjac'+R);
        %U=(pmat*Cjac')/LL;        
        K = (pmat*Cjac')/(Cjac*pmat*Cjac'+R);

        %state correction
        xc(:,i) = xp(end,:)' + K*(data(i,:)' - hobs);
        %xc(:,i) = xp(end,:)'+U*(LL'\(data(i,:)'-hobs));
        
        %covariance correction
        pc(:,:,i) = (eye(L)-K*Cjac)*pmat;
        %pc(:,:,i) = pmat-U*U';
        
        %standard deviation calculation
        sd(i,:) = (sqrt(diag(pc(:,:,i))))';
    end
else % run extended filter with AD jacobian
    for i = 2:N;
        %PREDICTION:
        %predicton of state dynamics

        [tp, xp] = ode45(model,[time(i-1) time(i)],xc(:,i-1)',options,q);

        %initial condition for the covariance prediction
        initP = reshape(pc(:,:,i-1),1,L^2);

        %calculates the jacobian for use in the covariance prediction step
        %covariance differential equation predictor
        [tp, pp] = ode15s(@cov_AD,[time(i-1) time(i)],initP,options,model,xc,L,Q,q);
        pmat = reshape(pp(end,:),L,L);

        %jacobian of the observation function
        xnew = myAD(xp(end,:));
        obold = feval(obser,xnew);
        Cjac = getderivs(obold);
        hobs = getvalue(obold);

        %CORRECTION:
        %kalman gain
        %[sm, lam, xn] = svd(Cjac*pmat*Cjac'+R);
        %invmat = xn*inv(lam)*sm';
        %K = pmat*Cjac'*invmat;
        %LL=chol(Cjac*pmat*Cjac'+R);
        %U=(pmat*Cjac')/LL;
        K = (pmat*Cjac')/(Cjac*pmat*Cjac'+R); 

        %state correction
        xc(:,i) = xp(end,:)' + K*(data(i,:)' - hobs);
        %xc(:,i) = xp(end,:)'+U*(LL'\(data(i,:)'-hobs));
        
        %covariance correction
        pc(:,:,i) = (eye(L)-K*Cjac)*pmat;
        %pc(:,:,i) = pmat-U*U';
        
        %standard deviation calculation
        sd(i,:) = (sqrt(diag(pc(:,:,i))))';
    end
end
out.xfilter = xc';
out.time = time;
out.P = pc;
out.stdev = sd;


function dp = cov_AD(t,p,model,xc,L,Q,q)
%   AD covariance differential equation 
    xold = myAD(xc(:,i-1));
    fold = feval(model,t,xold);
    A = getderivs(fold);
    pnew = reshape(p,L,L);
    dpdt = pnew*A'+A*pnew + Q;
    dp = reshape(dpdt,L^2,1);
end

function dp = cov_fd(t,p,model,xc,L,Q,q)
%   FD covariance differential dquation
    xold = xc(:,i-1);
    A = diffjac(model, xold,t,q);
    pnew = reshape(p,L,L);
    dpdt = pnew*A' + A*pnew + Q;
    dp = reshape(dpdt,L^2,1);
end


function jac = diffjac(f, x, t, q, tol)
%
% DIFFJAC
%   this calculates a finite difference jacobian for a given
%   function given a value, x, at a set tolerance level, tol
%
%   INPUT:
%       f: function handle
%       x: value at which jacobian is calculated at
%       tol: tolerance of accuracy you want jacobian calculated
%   
%   OUTPUT:
%       jac: jacobian matrix calculated at x at tolerance, tol
%
%   This code comes with no warranty or guarantee of any kind
%
%   June 11th, 2010
%   Brett J. Matzuka, NCSU

if nargin < 5
    tol = 2e-2;
end

x=x(:);

f0 = feval(f,t,x,q);

n = length(x);
jac = zeros(length(f0),n);

for j = 1:n
    zz = zeros(n,1);
    zz(j) = 1;
    jac(:,j) = dirder(x,zz,f,f0,t,tol);
end

function z = dirder(x,w,f,f0,t,tol)
% Compute a finite difference directional derivative.
% Approximate f'(x) w
% 
% C. T. Kelley, April 1, 2003
%
% This code comes with no guarantee or warranty of any kind.
%
% function z = dirder(x,w,f,f0)
%
% inputs:
%           x, w = point and direction
%           f = function
%           f0 = f(x), in nonlinear iterations
%                f(x) has usually been computed
%                before the call to dirder

% Hardwired difference increment.
epsnew = tol;
%
n = length(x);
%
% scale the step
%
if norm(w) == 0
    z = zeros(n,1);
return
end
%
% Now scale the difference increment.
%
xs=(x'*w)/norm(w);
if xs ~= 0.d0
     epsnew=epsnew*max(abs(xs),1.d0)*sign(xs);
end
epsnew=epsnew/norm(w);
%
% del and f1 could share the same space if storage
% is more important than clarity.
%
del = x+epsnew*w;
f1 = feval(f,t,del,q);
z = (f1 - f0)/epsnew;
% end function dirder
end
%end function diffjac
end

%end of Extended filter code
end

    

    