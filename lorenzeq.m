function dx=lorenzeq(t,x,q)

sigma=q(1);
rho=q(2);
beta=q(3);

dx=zeros(3,1);

dx=[sigma*(x(2)-x(1));
    x(1)*(rho-x(3))-x(2);
    x(1)*x(2)-beta*x(3)];