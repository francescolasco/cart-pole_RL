function sp = dinamica(s, m, M, L, g, a, Ts)

% transform action in acceleration
switch a
    case 1
        u = 30;
    case 2
        u = -30;
end

mmodel = @(t,x,u) model(s,m,M,L,g,u); 
[~, x] = ode45(@(t, x) mmodel(t, s, u), [0,Ts], s);
sp = x(end,:);

% % we should integrate continuous-time dynamics
% Ts = 1; % sampling time
% % integrate differential equations
% odefun = @(t,s) [s(2); 0.001*a - 0.0025*cos(3*x(1))];
% [t, st] = ode45(odefun, [0, Ts], s);
% % get next states as the last one
% sp = st(end,:)';

% to simplify, we use the forward Euler discretization
% xp = x + v;
% vp = v + 0.001*acc - 0.0025*cos(3*x);
%xp = x + tt*v;
%vp = v + tt*acc;  

% saturate the next state
% sp(1) = max(min(sp(1), X(2)),X(1));
% sp(2) = max(min(sp(2), V(2)),V(1));
% sp(3) = max(min(sp(3), THETA(2)),THETA(1));
% sp(4) = max(min(sp(4), OMEGA(2)),OMEGA(1));

% Calcolo il reward
% r = - ((sp(3)-pi) / (pi/4))^2;
% if (sp(3)-pi)^2 < 0.05
%     r = 0;
% else
%     r = -10;
% end



% Condizione sullo stato terminale: purtroppo in questo modo si mantiene
% una velocità lineare a regime non nulla, inserendo nel vincolo terminale
% anche una velocità < epsilon il modello non arriva mai a convergenza
% Provare anche (0.01*sp(1)^2 + sp(2)^2 + (sp(3)-pi)^2 + sp(4)^2) < 0.01

