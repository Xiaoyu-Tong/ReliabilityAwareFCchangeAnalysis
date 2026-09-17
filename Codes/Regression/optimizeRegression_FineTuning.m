function [W,beta,nNetD,L_history,L_sub_FT] = optimizeRegression_FineTuning(X01,X02,X11,X12,Y,W,beta,M,Mbeta,Kpred,L_history,max_iter,tol)
% 
% *** This function implements the optimization procedure as described in
% Box 2.3 in the Appendix of accompanying manuscript.
%
%
% Inputs:
% X01 -- run 1 of baseline FC (N x D)
% X02 -- run 2 of baseline FC (N x D), X01 and X02 are interchangable
% X11 -- run 1 of week 1 FC (N x D)
% X12 -- run 2 of week 1 FC (N x D), X11 and X12 are interchangable
% Y --  data vector for prediction target (N x 1)
% W -- continuous sparsification-based estimate of W (D x PD)
% beta --  continuous sparsification-based estimate of beta (PD x 1)
% M -- continuous sparsification-based estimate of M (D x PD)
% Mbeta --  continuous sparsification-based estimate of Mbeta (PD x 1)
% Kpred -- the relative weight of prediction terms
% max_iter -- max number of iterations
% tol -- error tolerance (stop sign for optimization)

% *** Note, X's should be normalized so that regularization is fair
% for each features. Y is also better to be normalized **
%
% Output:
% W -- dimension loadings for differential dimensions
% beta -- predictive feature weights of differential dimensions
% Others are for troubleshooting purposes

%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu

% tol = 10^-6;
iIter = 1;
L_new = L_history(end);
L = 2*L_new;
L_sub_FT = [];
% L_history = L_new;
convergeThres = 5;
convergeFlag = 0;

% Dichotomization (Mask Dichotomization in Box 2.3)
H = @(x) x>=0; % Heaviside step function

hidxd = H(M);
hidxbd = H(Mbeta);
hidxbd = (sum(hidxd)~=0)' & hidxbd;

W = W(:,hidxbd);
beta = beta(hidxbd);
hidxd = hidxd(:,hidxbd);

mu_dim = 0.01;
mu_pred = 0.01;

Wstar = W.*hidxd;
Ytilde = (X11+X12-X01-X02)/2*Wstar*beta;

Gbar = (X11+X12+X01+X02)*Wstar/4;
V_XW = norm(X11*Wstar - Gbar,"fro")^2 + norm(X12*Wstar - Gbar,"fro")^2 ...
            + norm(X01*Wstar - Gbar,"fro")^2 + norm(X02*Wstar - Gbar,"fro")^2;
scaler = 2+2*Kpred*V_XW*(beta'*beta);

% initialization with instant optimum
G1 = (X11+X12)*Wstar/2 + Kpred*V_XW/scaler*(Y-Ytilde)*beta';
G0 = (X01+X02)*Wstar/2 - Kpred*V_XW/scaler*(Y-Ytilde)*beta';

while (convergeFlag <= convergeThres) && iIter < max_iter*10
    if L_new < tol
        break
    end
    L = L_new;
    
    % Calculate auxiliary variables as described in Box 2.3 (P_signal through L_SNR)
    Wstar = W.*hidxd;
    Gbar = (G1+G0)/2;
    noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
        norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
    signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
        norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
    L_SNR = noisePower/signalPower;
    Gres_noise = X11'*(X11*Wstar-G1) + X12'*(X12*Wstar-G1) + X01'*(X01*Wstar-G0) + X02'*(X02*Wstar-G0);
    Gres_signal = X11'*(X11*Wstar-Gbar) + X12'*(X12*Wstar-Gbar) + X01'*(X01*Wstar-Gbar) + X02'*(X02*Wstar-Gbar);
    Gkernel_dim = (Gres_noise-L_SNR*Gres_signal)/signalPower;
    
    % Gradient descents (updates of W and beta in Box 2.3) 
    beta_new = beta - mu_pred*2*Kpred*(G1-G0)'*((G1-G0)*beta-Y);
    W_new = W - mu_dim*Gkernel_dim;

    % instant optimal values of G (updates of V_XW and G's in Box 2.3)
    Wstar_new = W_new.*hidxd;
    Ytilde = (X11+X12-X01-X02)/2*Wstar_new*beta_new;
    Gbar = (X11+X12+X01+X02)*Wstar_new/4;
    V_XW = norm(X11*Wstar_new - Gbar,"fro")^2 + norm(X12*Wstar_new - Gbar,"fro")^2 ...
        + norm(X01*Wstar_new - Gbar,"fro")^2 + norm(X02*Wstar_new - Gbar,"fro")^2;
    scaler = 2+2*Kpred*V_XW*(beta_new'*beta_new);

    G1_new = (X11+X12)*Wstar_new/2 + Kpred*V_XW/scaler*(Y-Ytilde)*beta_new';
    G0_new = (X01+X02)*Wstar_new/2 - Kpred*V_XW/scaler*(Y-Ytilde)*beta_new';
    
    % new loss function
    Gbar_new = (G1_new + G0_new)/2;
    noisePower_new = norm(G1_new - X11*Wstar_new,"fro")^2 + norm(G1_new - X12*Wstar_new,"fro")^2 +...
        norm(G0_new - X01*Wstar_new,"fro")^2 + norm(G0_new - X02*Wstar_new,"fro")^2;
    signalPower_new = norm(Gbar_new - X11*Wstar_new,"fro")^2 + norm(Gbar_new - X12*Wstar_new,"fro")^2 +...
        norm(Gbar_new - X01*Wstar_new,"fro")^2 + norm(Gbar_new - X02*Wstar_new,"fro")^2;
    L_SNR_new = noisePower_new/signalPower_new;

    L_new = L_SNR_new + Kpred * norm(Y-(G1_new-G0_new)*beta_new,"fro")^2;

    if L_new > L && iIter > 1
        mu_dim = mu_dim/2;
        mu_pred = mu_pred/2;
%         disp('learning rate reduced')
    else    
        L_history = [L_history,L_new];
        W = W_new;
        beta = beta_new;
        G0 = G0_new;
        G1 = G1_new;
        iIter = iIter + 1;
    end 
    Wstar = W.*hidxd;
    Gbar = (G1+G0)/2;
    noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
        norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
    signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
        norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
    L_SNR = noisePower/signalPower;
    LP = Kpred * norm(Y-(G1-G0)*beta,"fro")^2;
    L_sub_FT = [L_sub_FT,[L_SNR;LP]];

    if abs(L-L_new) > tol
        convergeFlag = 0;
    else
        convergeFlag = convergeFlag + 1;
    end
end

W = W.*hidxd;
nNetD = sum(hidxbd);

end









