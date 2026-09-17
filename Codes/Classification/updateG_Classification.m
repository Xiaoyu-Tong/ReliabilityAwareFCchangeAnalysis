function [G1_new,G0_new] = updateG_Classification(X11,X12,X01,X02,Wstar,gammaD,G1,G0,Y,Kclass)

% This function utilizes an efficient procedure to solve for the numerical
% solution of G1 and G0, in order to improve the efficiency and stability
% of overall loss function optimization. For the underlying mathematical
% analyses, see Appendix of the accompanying paper.

ftol = 10^-10;
options = optimset('TolX',ftol);

G1_new = zeros(size(G1));
G0_new = zeros(size(G0));
nSub = length(Y);

% The procedure for numerical solution of G1 and G0
% 1. Calculate sum of G1 and G0
Gsum = (X11+X12+X01+X02)*Wstar/2;
% 2. Calculate the variance of X*W (the denominator)
Var_XW = norm(X11*Wstar - Gsum/2,"fro")^2 + norm(X12*Wstar - Gsum/2,"fro")^2 ...
    + norm(X01*Wstar - Gsum/2,"fro")^2 + norm(X02*Wstar - Gsum/2,"fro")^2;
% The following procedure is implemented on each iSub = 1:nSub
% Auxiliary variables
Ghat = (X11+X12-X01-X02)*Wstar/2;
taskCoef = Kclass*Var_XW*(gammaD'*gammaD)/2;
for iSub = 1:nSub
    % 3. Find numerical solution of eta0
    Ghati = Ghat(iSub,:);
    p_hat = Ghati*gammaD;
    f = @(eta) (exp(p_hat+eta)+1)*(eta+taskCoef*(1-Y(iSub)))-taskCoef;
    % Calculate lower & upper bounds of roots tofacilitate root-solving
    rootLB = taskCoef*(Y(iSub)-1);
    rootUB = taskCoef*Y(iSub);
    % Ensure stability (get rid of infinity)
    while isinf(f(rootLB)) 
        rootLB = (rootLB + rootUB)/2;
    end
    while isinf(f(rootUB)) 
        rootUB = (rootLB + rootUB)/2;
    end
    eta0 = fzero(f,[rootLB,rootUB],options);
    % 4. Calculate G1_new and G0_new
    if gammaD'*gammaD
        res = gammaD'*eta0/(gammaD'*gammaD)/2;
    else
        res = 0;
    end
    G1_new(iSub,:) = (X11(iSub,:)+X12(iSub,:))*Wstar/2 + res;
    G0_new(iSub,:) = (X01(iSub,:)+X02(iSub,:))*Wstar/2 - res;
end


end

