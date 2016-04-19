function dx = lorenzeq_aug(t,x,q)

%dx = zeros(6,1);

dx = [x(4)*(x(2)-x(1));
    x(1)*(x(5)-x(3))-x(2);
    x(1)*x(2) - x(6)*x(3);
    0;
    0;
    0;];