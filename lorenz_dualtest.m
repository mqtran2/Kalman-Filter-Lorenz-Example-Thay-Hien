% lorenz equations example for dual filtering
close all
clear all

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
Rr = 0.8*eye(3);        % parameter process covariance
%Rr(1,1) = 0.01;
%Rr(2,2) = 0.01;
%Rr(3,3) = 0.01;
q0=[5, 15, 3/4];
param = [1 2 3];

load lorenzdata
truth=truth';
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

out = dualsrukf(@lorenzeq,obsfun,data,t,model0,R,V,P0x,P0q,Re,Rr,q0,param);
out1 = dualukf(@lorenzeq,obsfun,data,t,model0,R,V,P0x,P0q,Re,Rr,q0,param);
out2 = dualsrckf(@lorenzeq,obsfun,data,t,model0,R,V,P0x,P0q,Re,Rr,q0,param);
out3 = dualckf(@lorenzeq,obsfun,data,t,model0,R,V,P0x,P0q,Re,Rr,q0,param);
out4=srukf(@lorenzeq_aug,obsfun,data,t,x0,R3,V3,P0x2,[]);
%out5 = augSEnKF(@lorenzeq,obsfun,data,t,model0,R,V,P0x,q0,P0q,Rr,50,param);

time = t;

disp('SR-UKF')
out.qfilter(end,:)
disp('UKF')
out1.qfilter(end,:)
disp('SR-CKF')
out2.qfilter(end,:)
disp('CKF')
out3.qfilter(end,:)
disp('Joint SR-UKF')
out4.xfilter(end,3:5)
disp('what we want: [10 28 2.6667]')

figure
plot(time,truth(:,1),'b+')
hold on
plot(time,out.xfilter(:,1),'b-.')
plot(time,out1.xfilter(:,1),'g--')
plot(time,out2.xfilter(:,1),'r-s')
plot(time,out3.xfilter(:,1),'m')
plot(time,out4.xfilter(:,1),'y')
plot(time,model(1,:),'k')
title('dual estimation - position')
legend('data','dual SRUKF','Dual UKF', 'Dual SR-CKF','Dual CKF')
xlabel('time')

figure
plot(time,truth(:,2),'b+')
hold on
plot(time,out.xfilter(:,2),'b-.')
plot(time,out1.xfilter(:,2),'g--')
plot(time,out2.xfilter(:,2),'r-s')
plot(time,out3.xfilter(:,2),'m')
plot(time,out4.xfilter(:,2),'y')
plot(time,model(2,:),'k')
title('dual estimation - velocity')
legend('data','dual SRUKF','Dual UKF', 'Dual SR-CKF','Dual CKF')
xlabel('time')

figure
plot(time,out.qfilter(:,1))
hold on
plot(time,out.qfilter(:,2),'g')
plot(time,out.qfilter(:,3),'r')
plot(time,out.qfilter(:,1)+out.tsdq(:,1),'b--')
plot(time,out.qfilter(:,1)-out.tsdq(:,1),'b--')
plot(time,out.qfilter(:,2)-out.tsdq(:,2),'g--')
plot(time,out.qfilter(:,2)+out.tsdq(:,2),'g--')
plot(time,out.qfilter(:,3)+out.tsdq(:,3),'r--')
plot(time,out.qfilter(:,3)-out.tsdq(:,3),'r--')

% figure
% plot(time,out.tsdq)

figure
plot(time,out.qfilter)
xlabel('time')
title('SRUKF')
figure
plot(time,out1.qfilter)
xlabel('time')
title('UKF')
figure
plot(time,out2.qfilter)
xlabel('time')
title('SRCKF')
figure
plot(time,out3.qfilter)
xlabel('time')
title('CKF')
figure
plot(time,out4.xfilter(:,4:6))
xlabel('time')
title('joint ukf')