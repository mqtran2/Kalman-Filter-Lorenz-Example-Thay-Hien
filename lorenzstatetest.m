% lorenz state test
load lorenzdata
truth=truth';
model=model';
data = truth(:,1);

M = zeros(3,6);
M(1:3,1:3) = eye(3);

R = 0.1*eye(1);
V = 0.01*eye(3);
Vq = 0.5*eye(3);
P0x = 10*eye(3);
P0q = 5*eye(3);
qtruth = [10,28,8/3];
q0 = [5,21,1/3];
model0=[0.9, 1,1.1]';
model0j = [0.9 1, 1.1, 5,21,1/3]';
dynfun=@lorenzeq;
obsfun1=eye(3);
obsfune=@(t,y,q) [y(1);];
obsfun=@(y) [y(1)];
n=length(model0);

R2 = 0.1*eye(1);
V2 = 0.01*eye(6);
V2(4,4) = 0.5;
V2(5,5) = 0.5;
V2(6,6) = 0.5;
P0x2 = 5*eye(6);

param = [1,2,3];

% out=EnKF(@lorenzeq,obsfun1,data,t,model0,R,V,P0x,50,qtruth);
% 
time=t;
% 
% figure
% plot(time,model(:,1),'b')
% hold on
% plot(time,truth(:,1),'rs')
% plot(time,out.xfilter(:,1),'g-*')
% plot(time,out.xfilter(:,1)+out.tsd(:,1),'g:')
% plot(time,out.xfilter(:,1)-out.tsd(:,1),'g:')
% legend('truth','data','filter','3 std')
% title('EnKF - state 1')
% xlabel('time')
% ylabel('x(1)')
% 
% figure
% plot(time,model(:,2),'b')
% hold on
% plot(time,truth(:,2),'rs')
% plot(time,out.xfilter(:,2),'g-*')
% plot(time,out.xfilter(:,2)+out.tsd(:,2),'g:')
% plot(time,out.xfilter(:,2)-out.tsd(:,2),'g:')
% legend('truth','data','filter','3 std')
% title('EnKF - state 2')
% xlabel('time')
% ylabel('x(2)')
% 
% figure
% plot(time,model(:,3),'b')
% hold on
% plot(time,truth(:,3),'rs')
% plot(time,out.xfilter(:,3),'g-*')
% plot(time,out.xfilter(:,3)+out.tsd(:,3),'g:')
% plot(time,out.xfilter(:,3)-out.tsd(:,3),'g:')
% legend('truth','data','filter','3 std')
% title('EnKF - state 3')
% xlabel('time')
% ylabel('x(3)')

% display('EAKF')
% tic
% outa = augEAKF(@lorenzeq,M,data,t,model0,R,V,P0x,q0,P0q,Vq,50,param);
% toc
% % display('SEnKF')
% % tic
% % outp = augSEnKF(@lorenzeq,obsfun,data,t,model0,R,V,P0x,q0,P0q,Vq,50,param,0);
% % toc
% disp('ETKF')
% tic
% outp2 = augETKF(@lorenzeq,obsfun,data,t,model0,R,V,P0x,q0,P0q,Vq,50,param);
% toc
% display('EnKF')
% tic
% out3 = augEnKF(@lorenzeq,M,data,t,model0,R,V,P0x,q0,P0q,Vq,50,param);
% toc
display('joint EKF')
tic
outl = ekbf(@lorenzeq_aug,obsfune,data,t,R2,V2,model0j,P0x2,[],0);
toc
display('dual ukf')
tic
outc = dualukf(@lorenzeq,obsfun,data,t,model0,R,V,P0x,P0q,R,Vq,q0,param);
toc
display('joint ukf')
tic
outj = srukf(@lorenzeq_aug,obsfun,data,t,model0j,R2,V2,P0x2,[]);
toc
display('dual ckf')
tic
outc2 = dualckf(@lorenzeq,obsfun,data,t,model0,R,V,P0x,P0q,R,Vq,q0,param);
toc
display('joint ckf')
tic
outj2 = ckbf(@lorenzeq_aug,obsfun,data,t,model0j,R2,V2,P0x2,[]);
toc

display('joint EKF')
((outl.xfilter(:,1)-model(:,1))'*(outl.xfilter(:,1)-model(:,1)) + (outl.xfilter(:,2)-model(:,2))'*(outl.xfilter(:,2)-model(:,2)) + (outl.xfilter(:,3)-model(:,3))'*(outl.xfilter(:,3)-model(:,3)))*(1/length(time))
(outl.xfilter(end,4)-10)*(outl.xfilter(end,4)-10)' + (outl.xfilter(end,5)-28)*(outl.xfilter(end,5)-28)' + (outl.xfilter(end,6)-8/3)*(outl.xfilter(end,6)-8/3)'
display('joint ukf')
((outj.xfilter(:,1)-model(:,1))'*(outj.xfilter(:,1)-model(:,1)) + (outj.xfilter(:,2)-model(:,2))'*(outj.xfilter(:,2)-model(:,2)) + (outj.xfilter(:,3)-model(:,3))'*(outj.xfilter(:,3)-model(:,3)))*(1/length(time))
(outj.xfilter(end,4)-10)*(outj.xfilter(end,4)-10)' + (outj.xfilter(end,5)-28)*(outj.xfilter(end,5)-28)' + (outj.xfilter(end,6)-8/3)*(outj.xfilter(end,6)-8/3)'
display('joint ckf')
((outj2.xfilter(:,1)-model(:,1))'*(outj2.xfilter(:,1)-model(:,1)) + (outj2.xfilter(:,2)-model(:,2))'*(outj2.xfilter(:,2)-model(:,2)) + (outj2.xfilter(:,3)-model(:,3))'*(outj2.xfilter(:,3)-model(:,3)))*(1/length(time))
(outj2.xfilter(end,4)-10)*(outj2.xfilter(end,4)-10)' + (outj2.xfilter(end,5)-28)*(outj2.xfilter(end,5)-28)' + (outj2.xfilter(end,6)-8/3)*(outj2.xfilter(end,6)-8/3)'
display('dual ukf')
((outc.xfilter(:,1)-model(:,1))'*(outc.xfilter(:,1)-model(:,1)) + (outc.xfilter(:,2)-model(:,2))'*(outc.xfilter(:,2)-model(:,2)) + (outc.xfilter(:,3)-model(:,3))'*(outc.xfilter(:,3)-model(:,3)))*(1/length(time))
(outc.qfilter(end,1)-10)*(outc.qfilter(end,1)-10) + (outc.qfilter(end,2)-28)'*(outc.qfilter(end,2)-28) + (outc.qfilter(end,3)-8/3)'*(outc.qfilter(end,3)-8/3)
display('dual ckf')
((outc2.xfilter(:,1)-model(:,1))'*(outc2.xfilter(:,1)-model(:,1)) + (outc2.xfilter(:,2)-model(:,2))'*(outc2.xfilter(:,2)-model(:,2)) + (outc2.xfilter(:,3)-model(:,3))'*(outc2.xfilter(:,3)-model(:,3)))*(1/length(time))
(outc2.qfilter(end,1)-10)*(outc2.qfilter(end,1)-10) + (outc2.qfilter(end,2)-28)'*(outc2.qfilter(end,2)-28) + (outc2.qfilter(end,3)-8/3)'*(outc2.qfilter(end,3)-8/3)



figure
plot(time,model(:,1),'b')
hold on
plot(time,truth(:,1),'rs')
%plot(time,outp.xfilter(:,1),'g-*')
%plot(time,outp2.xfilter(:,1),'m')
%plot(time,out3.xfilter(:,1),'k')
plot(time,outc.xfilter(:,1),'m')
plot(time,outj.xfilter(:,1),'m--')
plot(time,outc2.xfilter(:,1),'g')
plot(time,outj2.xfilter(:,1),'g--')
plot(time,outl.xfilter(:,1),'c--')
%plot(time,outp.xfilter(:,1)+outp.tsdx(:,1),'g:')
%plot(time,outp.xfilter(:,1)-outp.tsdx(:,1),'g:')
%plot(time,outp2.xfilter(:,1)+outp2.tsdx(:,1),'m:')
%plot(time,outp2.xfilter(:,1)-outp2.tsdx(:,1),'m:')
%legend('truth','data','ETKF','EnKF','dual ukf','joint ukf','dual ckf','joint ckf','3 std')
legend('truth','data','dual ukf','joint ukf','dual ckf','joint ckf','joint ekf')
title('filters - state 1')
xlabel('time')
ylabel('x(1)')

figure
plot(time,model(:,2),'b')
hold on
plot(time,truth(:,2),'rs')
%plot(time,outp.xfilter(:,2),'g-*')
%plot(time,outp2.xfilter(:,2),'m')
%plot(time,out3.xfilter(:,2),'k')
plot(time,outc.xfilter(:,2),'m')
plot(time,outj.xfilter(:,2),'m--')
plot(time,outc2.xfilter(:,2),'g')
plot(time,outj2.xfilter(:,2),'g--')
plot(time,outl.xfilter(:,2),'c--')
%plot(time,outp.xfilter(:,2)+outp.tsdx(:,2),'g:')
%plot(time,outp.xfilter(:,2)-outp.tsdx(:,2),'g:')
%plot(time,outp2.xfilter(:,2)+outp2.tsdx(:,2),'m:')
%plot(time,outp2.xfilter(:,2)-outp2.tsdx(:,2),'m:')
%legend('truth','data','ETKF','EnKF','dual ukf','joint ukf','dual ckf','joint ckf','3 std')
legend('truth','data','dual ukf','joint ukf','dual ckf','joint ckf','joint ekf')
title('filters - state 2')
xlabel('time')
ylabel('x(2)')

figure
plot(time,model(:,3),'b')
hold on
plot(time,truth(:,3),'rs')
%plot(time,outp.xfilter(:,3),'g-*')
%plot(time,outp2.xfilter(:,3),'m')
%plot(time,out3.xfilter(:,3),'k')
plot(time,outc.xfilter(:,3),'m')
plot(time,outj.xfilter(:,3),'m--')
plot(time,outc2.xfilter(:,3),'g')
plot(time,outj2.xfilter(:,3),'g--')
plot(time,outl.xfilter(:,3),'c--')
%plot(time,outp.xfilter(:,3)+outp.tsdx(:,3),'g:')
%plot(time,outp.xfilter(:,3)-outp.tsdx(:,3),'g:')
%plot(time,outp2.xfilter(:,3)+outp2.tsdx(:,3),'m:')
%plot(time,outp2.xfilter(:,3)-outp2.tsdx(:,3),'m:')
%legend('truth','data','ETKF','EnKF','dual ukf','joint ukf','dual ckf','joint ckf','3 std')
legend('truth','data','dual ukf','joint ukf','dual ckf','joint ckf','joint ekf')
title('Filters - state 3')
xlabel('time')
ylabel('x(3)')


% figure
% plot(time,outj2.xfilter(:,4:6))
% hold on
% plot(time,outj2.xfilter(:,4:6)+outj2.tsd(:,4:6),':')
% plot(time,outj2.xfilter(:,4:6)-outj2.tsd(:,4:6),':')
% xlabel('time')
% ylabel('parameter estimates')
% legend('10','28','8/3')
% title('joint srUKF')
% figure
% plot(time,outc.qfilter)
% hold on
% plot(time,outc.qfilter+outc.tsdq,':')
% plot(time,outc.qfilter-outc.tsdq,':')
% xlabel('time')
% ylabel('parameter estimates')
% legend('10','28','8/3')
% title('dual srUKF')
% 
% % figure
% % plot(time,outp.qfilter)
% % hold on
% % plot(time,outp.qfilter+outp.tsdq,':')
% % plot(time,outp.qfilter-outp.tsdq,':')
% % xlabel('time')
% % ylabel('parameter estimates')
% % legend('10','28','8/3')
% % title('SEnKF')
% 
% figure
% hold on
% plot(time,outp2.qfilter)
% plot(time,outp2.qfilter+outp2.tsdq,':')
% plot(time,outp2.qfilter-outp2.tsdq,':')
% xlabel('time')
% ylabel('parameter estimates')
% legend('10','28','8/3')
% title('ETKF')
% figure
% hold on
% plot(time,out3.qfilter)
% plot(time,out3.qfilter+outp2.tsdq,':')
% plot(time,out3.qfilter-outp2.tsdq,':')
% xlabel('time')
% ylabel('parameter estimates')
% legend('10','28','8/3')
% title('EnKF')
% figure
% plot(time,outj22.xfilter(:,4:6))
% hold on
% plot(time,outj22.xfilter(:,4:6)+outj22.tsd(:,4:6),':')
% plot(time,outj22.xfilter(:,4:6)-outj22.tsd(:,4:6),':')
% xlabel('time')
% ylabel('parameter estimates')
% legend('10','28','8/3')
% title('joint srCKF')
% figure
% plot(time,outc2.qfilter)
% hold on
% plot(time,outc2.qfilter+outc2.tsdq,':')
% plot(time,outc2.qfilter-outc2.tsdq,':')
% xlabel('time')
% ylabel('parameter estimates')
% legend('10','28','8/3')
% title('dual srCKF')

figure
subplot(2,1,1)
plot(time,outc.qfilter,'--')
hold on
plot(time,outc2.qfilter,'-.')
plot(time,outc.qfilter+outc.tsdq,':')
plot(time,outc.qfilter-outc.tsdq,':')
plot(time,outc2.qfilter+outc2.tsdq,':')
plot(time,outc2.qfilter-outc2.tsdq,':')
xlabel('time')
ylabel('parameter estimates')
legend('10; 7','28; 21','8/3; 1')
title('dual filters')
subplot(2,1,2)
plot(time,outj.xfilter(:,4:6),'--')
hold on
plot(time,outj2.xfilter(:,4:6),'-.')
plot(time,outl.xfilter(:,4:6))
plot(time,outj.xfilter(:,4:6)+outj.tsd(:,4:6),':')
plot(time,outj.xfilter(:,4:6)-outj.tsd(:,4:6),':')
plot(time,outj2.xfilter(:,4:6)+outj2.tsd(:,4:6),':')
plot(time,outj2.xfilter(:,4:6)-outj2.tsd(:,4:6),':')
plot(time,outl.xfilter(:,4:6)+outl.sd(:,4:6),':')
plot(time,outl.xfilter(:,4:6)-outl.sd(:,4:6),':')
xlabel('time')
ylabel('parameter estimates')
legend('10; 7','28; 21','8/3; 1')
title('joint filters')

