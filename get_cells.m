function [cellX, cellV, cellTHETA, cellOMEGA] = get_cells(X, V, THETA, OMEGA, M, N)

% get size of grids on the four axes
wx = (X(2) - X(1)) / M;
wv = (V(2) - V(1)) / M;
wtheta = (THETA(2) - THETA(1)) / M;
womega = (OMEGA(2) - OMEGA(1)) / M;

% construct grids for x and v
xgrid = linspace(X(1) - wx, X(2), M + 2);
vgrid = linspace(V(1) - wv, V(2), M + 2);
thetagrid = linspace(THETA(1) - wtheta, THETA(2), M + 2);
omegagrid = linspace(OMEGA(1) - womega, OMEGA(2), M + 2);

% select displacement
%displacement = [3; 1];
displacement = [3; 1; 3; 1];
% SCEGLIERE MEGLIO I VALORI DEL DISPLACEMENT
% normalize displacement to 1
displacement = displacement/max(displacement);

% define movement between cells
mx = wx/N*displacement(1);
mv = wv/N*displacement(2);
mtheta = wtheta/N*displacement(3);
momega = womega/N*displacement(4);

% construct cells separately for x and v
cellX = zeros(N, M+2);
cellV = zeros(N, M+2);
cellTHETA = zeros(N, M+2);
cellOMEGA = zeros(N, M+2);

% the first grid is the one already built
cellX(1,:) = xgrid;
cellV(1,:) = vgrid;
cellTHETA(1,:) = thetagrid;
cellOMEGA(1,:) = omegagrid;
% the others are obtained using the movement defined above
for i = 2:N
    cellX(i,:) = xgrid + mx*(i-1);
    cellV(i,:) = vgrid + mv*(i-1);
    cellTHETA(i,:) = thetagrid + mtheta*(i-1);
    cellOMEGA(i,:) = omegagrid + momega*(i-1);
end