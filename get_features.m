function Fac = get_features(s, cellX, cellV, M, N)

% get state
x = s(1);
v = s(2);

% features active for each grid
FA = zeros(N,1);
% active components of the feature vector
Fac = zeros(N,1);

for i = 1:N
    % get the belonging cells by comparing the state with the extrema
    indx = find(x >= cellX(i, 1:end-1) ...
        & x <= cellX(i, 2:end), 1, 'first');
    indv = find(v >= cellV(i, 1:end-1) ...
        & v <= cellV(i, 2:end), 1, 'first');
    % get cell index 
    FA(i) = sub2ind([M+1 M+1], indx, indv);
    % get active components
    Fac(i) = FA(i) + (i-1)*(M+1)^2;
end

% % dimension of the feature vector
% d = (M+1)^2*N;
% % feature vector
% X = zeros(d, 1);
% X(Xac) = 1;