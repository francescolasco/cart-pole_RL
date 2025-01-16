function Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N)

    % get state
    x = s(1);
    v = s(2);
    theta = s(3);
    omega = s(4);
    
    % features active for each grid
    F = zeros(N,1);
    % active components of the feature vector
    Fac = zeros(N,1);
    
    for i = 1:N
        % get the belonging cells by comparing the state with the extrema
        indx = find(x >= cellX(i, 1:end-1) & x <= cellX(i, 2:end), 1, 'first');
        indv = find(v >= cellV(i, 1:end-1) & v <= cellV(i, 2:end), 1, 'first');
        indtheta = find(theta >= cellTHETA(i, 1:end-1) & theta <= cellTHETA(i, 2:end), 1, 'first');
        indomega = find(omega >= cellOMEGA(i, 1:end-1) & omega <= cellOMEGA(i, 2:end), 1, 'first');
    
        % get cell index 
        F(i) = sub2ind([M+1 M+1 M+1 M+1], indx, indv, indtheta, indomega);
        % QUESTO CONTIENE GLI INDICI LINEARE (PER OGNI GRIGLIA) RELATIVI ALLO
        % STATO CORRENTE
    
        % get active components
        Fac(i) = F(i) + (i-1)*(M+1)^4;
        % QUESTO CONTIENE GLI STESSI INDICI MA SPIAZZATI DI (M+1)^4 CELLE PER
        % TENERE CONTO DEL FATTO CHE OGNI INDICE è RELATIVO AD UNA GRIGLIA
        % DIVERSA
    end

end
% % dimension of the feature vector
% d = (M+1)^2*N;
% % feature vector
% X = zeros(d, 1);
% X(Xac) = 1;