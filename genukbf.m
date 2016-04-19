function out=genukbf(rhs,obsfun,data,time,x0,R,V,P0,q)

% GENUKBF Generalized Unscented Kalman Bucy Filter.
%   GENUKBF(rhs,obsfun,data,time,x0,R,V,P0,q) implements the continuous
%   time unscented kalman filter, where:
%       rhs: the ode model equations, nonlinear
%       obsfun: the observation function, nonlinear
%       data: the data that is being filtered
%       time: when the data were collected
%       x0: initial filter state
%       R: observation covariance, constant.
%       V: process noise covariance, constant.
%       P0: initial filter covariance
%       q: model parameters, optional but needed, (can use [])
%   And the output structure is simply:
%       out.xfilter: the xfilter output, in rows
%       out.time: your time vector back again
%       out.P: cell array with filter covariance at each time
%       out.tsd: plus/minus three standard deviations.

if ischar(rhs)==1; rhs=str2func(rhs); end
if ischar(obsfun)==1; obsfun=str2func(obsfun); end

L=numel(x0); % this is the n from the comments above.
N=length(data);
dt=time(2)-time(1);

% need to assemble the weights, which are not state dependant.
[X W Wm]=sigmapoints(x0,P0);

% lets init the filter. using a cell to store each filter covmat.
P=zeros(L,L,N);
xfilter=zeros(L,N);
tsd=zeros(N,L);

% fill in the initial conditions
xfilter(:,1)=x0;
P(:,:,1)=P0;
tsd(1,:)=(sqrt(diag(P(:,:,1)))*3)';

for k=1:N-1
    xminus=xfilter(:,k);
    Pminus=P(:,:,k);

    % mean and covariance prediction

    [tplus,kplus]=ode45(@odewP,[time(k) time(k+1)],[xminus(:); Pminus(:)],...
        [],rhs,L,q,Wm,W,V);

    % weighted mean
    xminus=kplus(end,1:L)';

    % weighted covariance
    % Pminus=Xhat*W*Xhat'+V;
    Pminus=reshape(kplus(end,L+1:end),L,L)';
    % using reshape ==> have to transpose, but covar is spd..


    % update parts
    Xminus=sigmapoints(xminus,Pminus,1,2,3-L);

    Y=zeros(size(data,2),2*L+1);

    % i know there is a way to do this without a loop, but i have been
    % unable to figure it out.

    for j=1:2*L+1
        Y(:,j)=feval(obsfun,Xminus(:,j));
    end

    yminus=Y*Wm;

    Pyy=Y*W*Y'+R;
    Pxy=Xminus*W*Y'; % cross variance

    %     K=Pxy/Pyy; % apologies to dr. chu who said to never take an
    %     inverse

    LL=chol(Pyy);
    U=Pxy/LL;
    Pplus=Pminus-U*U';
    xplus=xminus+U*(LL'\(data(k+1,:)'-yminus));

    %     xplus=xminus+K*(data(k+1,:)'-yminus);
    %     Pplus=Pminus-K*Pyy*K';

    xfilter(:,k+1)=xplus;
    P(:,:,k+1)=Pplus;

    tsd(k+1,:)=(sqrt(diag(P(:,:,k+1)))*3)';
end

out.xfilter=xfilter';
out.time=time;
out.P=P;
out.tsd=tsd;

function dx=odewP(t,x,rhs,L,q,Wm,W,V)

% computes the dxdt, state.
xminus=x(1:L);
Pminus=reshape(x(L+1:end),L,L);

X=sigmapoints(xminus,Pminus);
Xhat=zeros(L,2*L+1);

% i know there is a way to do this without a loop, but i have been
% unable to figure it out.

for j=1:2*L+1
    Xhat(:,j)=feval(rhs,t,X(:,j),q); % SigmaProp
end

dxdt=Xhat*Wm;

Pdot=X*W*Xhat'+Xhat*W*X'+V;

dx=[dxdt(:); Pdot(:)];

function [X W wm wc sc] = sigmapoints(m, P, a, b, k, issqrt)

% this code written by Jacob Wasserman at MIT-LL. Thanks Jacob!

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

%P_sqrt = sqrtm(P);
if any(eig(P)<0)
    P_sqrt=real(sqrtm(P));
    disp('neg eval, but lets not worry about it');
else
    P_sqrt = chol(P)';
end
%P_sqrt = real(sqrtm(P));
X = repmat(m, 1, np) + sc*[zeros(n,1) P_sqrt -P_sqrt];

% Form the weights
L = a*a*(n+k) - n;
wm = [L/(n+L) repmat(1/(2*(n+L)), 1, 2*n)]';
wc = [(L/(n+L) + (1 - a*a + b)) repmat(1/(2*(n+L)), 1, 2*n)]';

tmp = eye(np) - repmat(wm, 1, np);
W = tmp*diag(wc)*tmp';