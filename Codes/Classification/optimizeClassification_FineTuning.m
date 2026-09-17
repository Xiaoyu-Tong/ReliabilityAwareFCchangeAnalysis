function [W,beta,nNetD,L_history,L_sub_FT] = optimizeClassification_FineTuning(X01,X02,X11,X12,G1,G0,Y,W,beta,M,Mbeta,Kclass,L_history,max_iter,tol)
% 
% *** This function implements the optimization procedure as described in
% Box 3.3 in the Appendix of accompanying manuscript.
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
% Kclass -- the relative weight of classification terms
% max_iter -- max number of iterations
% tol -- error tolerance (stop sign for optimization)

% *** Note, X's should be normalized so that regularization is fair
% for each features. Y is also better to be normalized **
%
% Output:
% W -- dimension loadings for differential dimensions
% beta -- classification feature weights of differential dimensions for target
% Others are for troubleshooting purposes

%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu

% Equivalent Kclass
nSub = length(Y);
Kclass = Kclass/nSub;

% tol = 10^-6;
iIter = 1;
L_new = L_history(end);
L = 2*L_new;
L_sub_FT = [];
% L_history = L_new;
convergeThres = 5;
convergeFlag = 0;

H = @(x) x>=0; % Heaviside step function

hidxd = H(M);
hidxbd = H(Mbeta);

mu_dim = 0.1;
mu_pred = 0.1;
% mu_G = 0.1;

while (convergeFlag <= convergeThres) && (iIter < max_iter*2)
    if L_new < tol
        break
    end
    L = L_new;

    % Calculate auxiliary variables as described in Box 3.3 (P_signal through L_SNR)
    Wstar = W.*hidxd;
    Gbar = (G1+G0)/2;
    noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
        norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
    signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
        norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
    L_SNRstar = noisePower/signalPower;
    Gres_noise = X11'*(X11*Wstar-G1) + X12'*(X12*Wstar-G1) + X01'*(X01*Wstar-G0) + X02'*(X02*Wstar-G0);
    Gres_signal = X11'*(X11*Wstar-Gbar) + X12'*(X12*Wstar-Gbar) + X01'*(X01*Wstar-Gbar) + X02'*(X02*Wstar-Gbar);
    Gkernel_dim = (Gres_noise-L_SNRstar*Gres_signal)/signalPower;

    G = G1 - G0;
    gammaD = beta.*hidxbd;

    % Gradient descents (updates of W and beta in Box 3.3) 
    W_new = W - mu_dim*Gkernel_dim.*hidxd;
    beta_new = beta + mu_pred*Kclass*G'*Y.*hidxbd;

    % instant optimal values of G (updates of V_XW and G's in Box 3.3)
    G1_new = zeros(size(G1));
    G0_new = zeros(size(G0));
    for iSub = 1:nSub
        Gi = G(iSub,:);
        Gbari = Gbar(iSub,:);
        pY = exp(Gi*gammaD);
        beta_new = beta_new - mu_pred*Kclass*pY/(pY+1)*Gi'.*hidxbd;
    end
    Wstar_new = W_new.*hidxd;
    gammaD_new = beta_new.*hidxbd;
    [G1_new,G0_new] = updateG_Classification(X11,X12,X01,X02,Wstar_new,gammaD_new,G1,G0,Y,Kclass);

    % new loss function
    G_new = G1_new - G0_new;
    Gbar_new = (G1_new + G0_new)/2;
    Lclass_new = -Y'*G_new*gammaD_new;
    for iSub = 1:nSub
        Gi_new = G_new(iSub,:);
        Lclass_new = Lclass_new + log(exp(Gi_new*gammaD_new)+1);
    end
    noisePower_new = norm(G1_new - X11*Wstar_new,"fro")^2 + norm(G1_new - X12*Wstar_new,"fro")^2 +...
        norm(G0_new - X01*Wstar_new,"fro")^2 + norm(G0_new - X02*Wstar_new,"fro")^2;
    signalPower_new = norm(Gbar_new - X11*Wstar_new,"fro")^2 + norm(Gbar_new - X12*Wstar_new,"fro")^2 +...
        norm(Gbar_new - X01*Wstar_new,"fro")^2 + norm(Gbar_new - X02*Wstar_new,"fro")^2;
    L_SNR_new = noisePower_new/signalPower_new;
    L_new = L_SNR_new + Kclass * Lclass_new;
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
    G = G1 - G0;
    gammaD = hidxbd.*beta;
    noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
        norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
    signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
        norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
    L_SNR = noisePower/signalPower;

    Lclass = -Y'*G*gammaD;
    for iSub = 1:nSub
        Gi = G(iSub,:);
        Lclass = Lclass_new + log(exp(Gi*gammaD)+1);
    end
    L_sub_FT = [L_sub_FT,[L_SNR;Lclass]];

    if abs(L-L_new) > tol
        convergeFlag = 0;
    else
        convergeFlag = convergeFlag + 1;
    end
end

beta = beta(hidxbd);
W = W(:,hidxbd);
hidxd = hidxd(:,hidxbd);
W = W.*hidxd;
idRealDim = logical(sum(W~=0,1));
W = W(:,idRealDim);
beta = beta(idRealDim);
nNetD = sum(hidxbd);

end









