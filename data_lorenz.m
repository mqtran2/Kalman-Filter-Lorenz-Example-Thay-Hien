function data_lorenz

time = [0.01:0.01:10];
dx = time(2)-time(1);
qtruth=[10,28,8/3];
model=[0.9 1 1.1]';
for i = 1:length(time)-1
    model(:,i+1) = rk4('lorenzeq',dx,time(i),model(:,i),qtruth);
end
t1 = time;
truth = model + 0.1*model.*randn(length(model),3)';

time = [10.01:0.01:30];
qtruth = [7,21,1];
model2 = [model(:,end)];
for i = 1:length(time)-1
    model2(:,i+1) = rk4('lorenzeq',dx,time(i),model2(:,i),qtruth);
end

t2 = time;
truth2 = model2 + 0.1*model2.*randn(length(model2),3)';

truth = [truth, truth2];
model = [model, model2]; 
t = [t1 t2];

figure
plot(t,truth,'r*')
hold on
plot(t,model,'b')
xlabel('time')
ylabel('state')
title('lorenz data creation')

save lorenzdatamix t truth model

function dx=rk4(rhs,h,t,x,q)

% fourth order explicit rk integrator

k1=feval(rhs,t,x,q);
k2=feval(rhs,t+h/2,x+h/2*k1,q);
k3=feval(rhs,t+h/2,x+h/2*k2,q);
k4=feval(rhs,t+h,x+h*k3,q);

dx=x+h/6*(k1+2*k2+2*k3+k4);