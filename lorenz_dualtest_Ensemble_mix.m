% lorenz equations example for dual filtering
%close all
%clear all

M = zeros(3,6);
M(1:3,1:3) = eye(3);
qtruth=[10,28,8/3];
model0=[0.9, 1, 1.1]';
dynfun=@lorenzeq;
obsfun=@(x) [x(1);x(2);x(3)];
n=length(model0);

R=0.1*eye(3);           % observation covariance
V=0.001*eye(3);           % Process covariance
P0x=10*eye(3);
P0q=5*eye(3);
Re=0.1*eye(3);         % parameter observation covariance
Rr = 0.5*eye(3);        % parameter process covariance
%Rr(1,1) = 0.01;
%Rr(2,2) = 0.01;
%Rr(3,3) = 0.01;
q0=[5, 15, 3/4];
param = [1 2 3];

load lorenzdatamix
truth=truth';
model=model';
%data = [truth(:,1); truth(:,2)];
data=truth;
%data = [truth(:,1),truth(:,3)];
R2=0.1*eye(3); %obser cov
V2 = 0.001*eye(6);  %process cov
x0 = [0.9 1 1.1 5 15 3/4]';
P0x2=10*eye(6);
R3 = 0.1*eye(3);
V3 = 0.001*eye(6);
V3(4,4) = 0.8;
V3(5,5) = 0.8;
V3(6,6) = 0.8;

disp('ETKF')
tic
out = augETKF(dynfun,obsfun,data,t,model0,R,V,P0x,q0,P0q,Rr,100,param,0);
toc
disp('truth')
param_truth = [7 21 1]
disp('ETKF')
out.qfilter(end,:)
(1/length(data))*((out.xfilter(:,1)-model(:,1))'*(out.xfilter(:,1)-model(:,1)) + (out.xfilter(:,2)-model(:,2))'*(out.xfilter(:,2)-model(:,2)) + (out.xfilter(:,3)-model(:,3))'*(out.xfilter(:,3)-model(:,3)))
(out.qfilter(end,1)-7)*(out.qfilter(end,1)-7)' + (out.qfilter(end,2)-21)*(out.qfilter(end,2)-21)' + (out.qfilter(end,3)-1)*(out.qfilter(end,3)-1)'
% disp('AEnKF')
% tic
% outa = augAEnKF(dynfun,M,data,t,model0,R,V,P0x,q0,P0q,Rr,100,param,0.9,0.5,0);
% toc
% disp('AEnKF')
% outa.qfilter(end,:)
% (outa.xfilter(:,1)-model(:,1))'*(outa.xfilter(:,1)-model(:,1)) + (outa.xfilter(:,2)-model(:,2))'*(outa.xfilter(:,2)-model(:,2)) + (outa.xfilter(:,3)-model(:,3))'*(outa.xfilter(:,3)-model(:,3))
% (outa.qfilter(end,1)-7)*(outa.qfilter(end,1)-7)' + (outa.qfilter(end,2)-21)*(outa.qfilter(end,2)-21)' + (outa.qfilter(end,3)-1)*(outa.qfilter(end,3)-1)'

disp('EnKF')
tic
outa = augEnKF(dynfun,M,data,t,model0,R,V,P0x,q0,P0q,Rr,100,param);
toc
disp('EnKF')
outa.qfilter(end,:)
(1/length(data))*((outa.xfilter(:,1)-model(:,1))'*(outa.xfilter(:,1)-model(:,1)) + (outa.xfilter(:,2)-model(:,2))'*(outa.xfilter(:,2)-model(:,2)) + (outa.xfilter(:,3)-model(:,3))'*(outa.xfilter(:,3)-model(:,3)))
(outa.qfilter(end,1)-7)*(outa.qfilter(end,1)-7)' + (outa.qfilter(end,2)-21)*(outa.qfilter(end,2)-21)' + (outa.qfilter(end,3)-1)*(outa.qfilter(end,3)-1)'


time = t;

figure
plot(time,model(:,1),'b')
hold on
plot(time,truth(:,1),'rs')
plot(time,out.xfilter(:,1),'y-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,outa.xfilter(:,1),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_1')

figure
plot(time,model(:,2),'b')
hold on
plot(time,truth(:,2),'rs')
plot(time,out.xfilter(:,2),'y-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,outa.xfilter(:,2),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_2')

figure
plot(time,model(:,3),'b')
hold on
plot(time,truth(:,3),'rs')
plot(time,out.xfilter(:,3),'y-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,outa.xfilter(:,3),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_3')

figure
subplot(2,1,1)
plot(time,model(:,1),'b')
hold on
plot(time,truth(:,1),'rs')
plot(time,out.xfilter(:,1),'y-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,outa.xfilter(:,1),'g--')
%plot(time,out1.xfilter(:,1),'m:')
legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_1')
title('State filtering - Ensemble filters')
subplot(2,1,2)
plot(time,model(:,3),'b')
hold on
plot(time,truth(:,3),'rs')
plot(time,out.xfilter(:,3),'y-.')
%plot(time,out2.xfilter(:,1),'c--')
%plot(time,out3.xfilter(:,1),'k')
plot(time,outa.xfilter(:,3),'g--')
%plot(time,out1.xfilter(:,1),'m:')
%legend('truth','data','ETKF','EnKF')
xlabel('time')
ylabel('x_3')



figure
subplot(3,1,1)
plot(time(1:1000),10,time(1001:end),7,'m-','linewidth',2)
hold on
plot(time,out.qfilter(:,1),'g--')
plot(time,outa.qfilter(:,1),'b--')
plot(10*ones(1,31),[10:-0.1:7],'m-','linewidth',2)
axis([0 30 0 20])
xlabel('time')
ylabel('\sigma')
title('Parameter Estimates for Ensemble filters')
subplot(3,1,2)
plot(time,out.qfilter(:,2),'g--')
hold on
plot(time,outa.qfilter(:,2),'b--')
plot(time(1:1000),28,'m-',time(1001:end),21,'m-','linewidth',2)
plot(10,[28:-0.1:21],'m-','linewidth',2)
axis([0 30 10 40])
xlabel('time')
ylabel('\rho')
legend('ETKF','EnKF','truth')
subplot(3,1,3)
plot(time(1:1000),8/3,time(1001:end),1,'m-','linewidth',4)
hold on
plot(time,out.qfilter(:,3),'g--')
plot(time,outa.qfilter(:,3),'b--')
plot(10*ones(1,17),[8/3:-0.1:1],'m-','linewidth',4)
axis([0 30 -2 8])
xlabel('time')
ylabel('\beta')

figure
plot(time,out.qfilter(:,1),'b')
hold on
plot(time,out.qfilter(:,2),'g')
plot(time,out.qfilter(:,3),'r')
plot(time,out.qfilter(:,1)+out.tsdq(:,1),'b--')
plot(time,out.qfilter(:,1)-out.tsdq(:,1),'b--')
plot(time,out.qfilter(:,2)-out.tsdq(:,2),'g--')
plot(time,out.qfilter(:,2)+out.tsdq(:,2),'g--')
plot(time,out.qfilter(:,3)+out.tsdq(:,3),'r--')
plot(time,out.qfilter(:,3)-out.tsdq(:,3),'r--')
% plot(time,outa.qfilter(:,1),'b:')
% plot(time,outa.qfilter(:,2),'g:')
% plot(time,outa.qfilter(:,3),'r:')
% plot(time,outa.qfilter(:,1)+outa.tsdq(:,1),'b.-')
% plot(time,outa.qfilter(:,1)-outa.tsdq(:,1),'b.-')
% plot(time,outa.qfilter(:,2)-outa.tsdq(:,2),'g.-')
% plot(time,outa.qfilter(:,2)+outa.tsdq(:,2),'g.-')
% plot(time,outa.qfilter(:,3)+outa.tsdq(:,3),'r.-')
% plot(time,outa.qfilter(:,3)-outa.tsdq(:,3),'r.-')
xlabel('time')
title('ETKF')
ylabel('Parameter Estimates')

figure
plot(time,outa.qfilter)
hold on
plot(time,outa.qfilter(:,2),'g')
plot(time,outa.qfilter(:,3),'r')
plot(time,outa.qfilter(:,1)+outa.tsdq(:,1),'b--')
plot(time,outa.qfilter(:,1)-outa.tsdq(:,1),'b--')
plot(time,outa.qfilter(:,2)-outa.tsdq(:,2),'g--')
plot(time,outa.qfilter(:,2)+outa.tsdq(:,2),'g--')
plot(time,outa.qfilter(:,3)+outa.tsdq(:,3),'r--')
plot(time,outa.qfilter(:,3)-outa.tsdq(:,3),'r--')
xlabel('time')
ylabel('Parameter Estimates')
title('AEnKF')