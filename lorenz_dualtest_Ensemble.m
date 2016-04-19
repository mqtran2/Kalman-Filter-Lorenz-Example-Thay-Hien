% lorenz equations example for dual filtering
% close all
% clear all

%M = zeros(3,6);
%M(1:3,1:3) = eye(3);
M = zeros(1,6);
M(1,1) = 1;
qtruth=[10,28,8/3];
model0=[0.9, 1, 1.1]';
dynfun=@lorenzeq;
obsfun=@(x) [x(1)];
n=length(model0);

R=0.1*eye(1);           % observation covariance
V=0.001*eye(3);           % Process covariance
RR=0.1*eye(3);
VV=0.001*eye(3);
P0x=10*eye(3);
P0q=5*eye(3);
Re=0.1*eye(1);         % parameter observation covariance
Rr = 0.001*eye(3);        % parameter process covariance
RRe = 0.1*eye(2);
RRr = 0.001*eye(3);
%Rr(1,1) = 0.01;
%Rr(2,2) = 0.01;
%Rr(3,3) = 0.01;
q0=[5, 21, 1/3];
param = [1 2 3];

load lorenzdata
truth=truth';
%data=[truth(:,1), truth(:,2)];
data = truth(:,1);
model=model';

R2=0.1*eye(3); %obser cov
V2 = 0.001*eye(6);  %process cov
x0 = [0.9 1 1.1 5 15 3/4]';
P0x2=10*eye(6);
R3 = 0.01*eye(3);
V3 = 0.01*eye(6);


%out = dualsrukf(@lorenzeq,obsfun,truth,t,model0,R,V,P0x,P0q,Re,Rr,q0,param);
%out1 = dualukf(@lorenzeq,obsfun,truth,t,model0,R,V,P0x,P0q,Re,Rr,q0,param);
%out2 = dualsrckf(@lorenzeq,obsfun,data,t,model0,RR,VV,P0x,P0q,RRe,RRr,q0,param);
%out3 = dualsrukf(@lorenzeq,obsfun,data,t,model0,RR,VV,P0x,P0q,RRe,RRr,q0,param);
%out4=srukf(@lorenzeq_aug,obsfun,truth,t,x0,R3,V3,P0x2,[]);
disp('ETKF')
tic
out = augETKF(@lorenzeq,obsfun,data,t,model0,R,V,P0x,q0,P0q,Rr,100,param,0);
toc
disp('EnKF')
tic
out5 = augEnKF(@lorenzeq,M,data,t,model0,R,V,P0x,q0,P0q,Rr,100,param,0);
toc
%out1 = nonlinaugEnKF(@lorenzeq,obsfun,data,t,model0,R,V,P0x,q0,P0q,Rr,100,param,0);

time = t;

disp('ETKF')
out.qfilter(end,:)
(out.xfilter(:,1)-model(:,1))'*(out.xfilter(:,1)-model(:,1)) + (out.xfilter(:,2)-model(:,2))'*(out.xfilter(:,2)-model(:,2)) + (out.xfilter(:,3)-model(:,3))'*(out.xfilter(:,3)-model(:,3))
(out.qfilter(end,1)-10)*(out.qfilter(end,1)-10)' + (out.qfilter(end,2)-28)*(out.qfilter(end,2)-28)' + (out.qfilter(end,3)-8/3)*(out.qfilter(end,3)-8/3)'

% disp('nonlinEnKF')
% out1.qfilter(end,:)
% disp('SR-CKF')
% out2.qfilter(end,:)
% disp('SR-UKF')
% out3.qfilter(end,:)
disp('EnKF')
% out4.xfilter(end,3:5)
out5.qfilter(end,:)
(out5.xfilter(:,1)-model(:,1))'*(out5.xfilter(:,1)-model(:,1)) + (out5.xfilter(:,2)-model(:,2))'*(out5.xfilter(:,2)-model(:,2)) + (out5.xfilter(:,3)-model(:,3))'*(out5.xfilter(:,3)-model(:,3))
(out5.qfilter(end,1)-10)*(out5.qfilter(end,1)-10)' + (out5.qfilter(end,2)-28)*(out5.qfilter(end,2)-28)' + (out5.qfilter(end,3)-8/3)*(out5.qfilter(end,3)-8/3)'

disp('what we want: [10 28 2.6667]')

figure
plot(time,model(:,1),'b')
hold on
plot(time,truth(:,1),'rs')
plot(time,out.xfilter(:,1),'k-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,out5.xfilter(:,1),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_1')

figure
plot(time,model(:,2),'b')
hold on
plot(time,truth(:,2),'rs')
plot(time,out.xfilter(:,2),'k-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,out5.xfilter(:,2),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_2')

figure
plot(time,model(:,3),'b')
hold on
plot(time,truth(:,3),'rs')
plot(time,out.xfilter(:,3),'k-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,out5.xfilter(:,3),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_3')

% figure
% plot(time,truth(:,2),'b+')
% hold on
% plot(time,out.xfilter(:,2),'r-.')
% % plot(time,out1.xfilter(:,2),'g--')
% plot(time,out2.xfilter(:,2),'c--')
% plot(time,out3.xfilter(:,2),'k')
% plot(time,out5.xfilter(:,2),'g')
% plot(time,out1.xfilter(:,2),'m:')
% plot(time,model(:,2),'y')
% title('dual estimation - velocity')
% legend('data','ETKF','SrCKF','SrUKF','EnKF','EnKF nonlin')
% xlabel('time')
% 
% figure
% plot(time,truth(:,3),'b+')
% hold on
% plot(time,out.xfilter(:,3),'r-.')
% % plot(time,out1.xfilter(:,2),'g--')
% plot(time,out2.xfilter(:,2),'c--')
% plot(time,out3.xfilter(:,2),'k')
% plot(time,out5.xfilter(:,3),'g')
% plot(time,out1.xfilter(:,3),'m:')
% plot(time,model(:,3),'y')
% title('dual estimation - velocity')
% legend('data','ETKF','SrCKF','SrUKF','EnKF','EnKF nonlin')
% xlabel('time')


% figure
% plot(time,out.tsdq)

% figure
% plot(time,out.qfilter)
% xlabel('time')
% title('SRUKF')
figure
plot(time,out.qfilter)
hold on
plot(time,out.qfilter(:,2),'g')
plot(time,out.qfilter(:,3),'r')
plot(time,out.qfilter(:,1)+out.tsdq(:,1),'b--')
plot(time,out.qfilter(:,1)-out.tsdq(:,1),'b--')
plot(time,out.qfilter(:,2)-out.tsdq(:,2),'g--')
plot(time,out.qfilter(:,2)+out.tsdq(:,2),'g--')
plot(time,out.qfilter(:,3)+out.tsdq(:,3),'r--')
plot(time,out.qfilter(:,3)-out.tsdq(:,3),'r--')
xlabel('time')
title('ETKF')
ylabel('Parameter Estimates')
% figure
% plot(time,out1.qfilter)
% hold on
% plot(time,out1.qfilter(:,2),'g')
% plot(time,out1.qfilter(:,3),'r')
% plot(time,out1.qfilter(:,1)+out1.tsdq(:,1),'b--')
% plot(time,out1.qfilter(:,1)-out1.tsdq(:,1),'b--')
% plot(time,out1.qfilter(:,2)-out1.tsdq(:,2),'g--')
% plot(time,out1.qfilter(:,2)+out1.tsdq(:,2),'g--')
% plot(time,out1.qfilter(:,3)+out1.tsdq(:,3),'r--')
% plot(time,out1.qfilter(:,3)-out1.tsdq(:,3),'r--')
% xlabel('time')
% title('EnKF nonlin')
% figure
% plot(time,out2.qfilter)
% xlabel('time')
% title('SRCKF')
% figure
% plot(time,out3.qfilter)
% xlabel('time')
% title('SRUKF')
figure
plot(time,out5.qfilter)
hold on
plot(time,out5.qfilter(:,2),'g')
plot(time,out5.qfilter(:,3),'r')
plot(time,out5.qfilter(:,1)+out5.tsdq(:,1),'b--')
plot(time,out5.qfilter(:,1)-out5.tsdq(:,1),'b--')
plot(time,out5.qfilter(:,2)-out5.tsdq(:,2),'g--')
plot(time,out5.qfilter(:,2)+out5.tsdq(:,2),'g--')
plot(time,out5.qfilter(:,3)+out5.tsdq(:,3),'r--')
plot(time,out5.qfilter(:,3)-out5.tsdq(:,3),'r--')
xlabel('time')
ylabel('Parameter Estimates')
title('EnKF')