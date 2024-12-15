function dx = model(x,m,M,L,g,d,u)

Sx = sin(x(3));
Cx = cos(x(3));
D = m*L*L*(M+m*(1-Cx^2));

mu = 0.45; % attrito

% dx(1,1) = x(2); % posizione                   
% dx(2,1) = (1/D)*(-m^2*L^2*g*Cx*Sx + m*L^2*(m*L*x(4)^2*Sx - d*x(2))) + m*L*L*(1/D)*u; % velocità
% dx(3,1) = x(4); % angolo 
% dx(4,1) = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*x(4)^2*Sx - d*x(2))) - m*L*Cx*(1/D)*u; % velocità angolare

dx(3,1) = x(4); 
dx(4,1) = (g*Sx + Cx*( (-u - m*L*x(4)^2*Sx + mu*sign(x(2)))/(M + m) ) - mu*x(4)/(m*L)) / (L*(4/3 - (m*Cx^2)/(m+M)));
dx(1,1) = x(2);                 
dx(2,1) = ( u + m*L*(x(4)^2*Sx - dx(4,1)*Cx) - mu*sign(x(2)) ) / (m + M); 