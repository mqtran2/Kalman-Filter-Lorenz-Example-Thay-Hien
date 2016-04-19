function data_creation

time = [0.01:0.01:5];
dx = time(2)-time(1);
qtruth=[100,4,60];
x0 = [5 4]';
model=x0;
for i = 1:length(time)-1
    model(:,i+1) = rk4('nonlinsp',dx,time(i),model(:,i),qtruth);
end
t = time;
truth = model + 0.1*model.*randn(length(model),2)';
data=truth;

save rkdata_final t data model qtruth x0


function dx=rk4(rhs,h,t,x,q)

% fourth order explicit rk integrator

k1=feval(rhs,t,x,q);
k2=feval(rhs,t+h/2,x+h/2*k1,q);
k3=feval(rhs,t+h/2,x+h/2*k2,q);
k4=feval(rhs,t+h,x+h*k3,q);

dx=x+h/6*(k1+2*k2+2*k3+k4);