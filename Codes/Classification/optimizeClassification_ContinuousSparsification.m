function [W,G0,G1,beta,M,Mbeta,alphaW,alphaBeta,L_history,L_sub] = optimizeClassification_ContinuousSparsification(X01,X02,X11,X12,G0,G1,Y,W,beta,Kclass,lambda_dim,lambda_class,max_iter,tol)
% 
% *** This function implements the optimization procedure as described in
% Box 3.2 in the Appendix of accompanying manuscript.
% This procedure is obtained from the gradient descents in Box 3.1 and
% subsequent derivations.
%
% Inputs:
% X01 -- run 1 of baseline FC (N x D)
% X02 -- run 2 of baseline FC (N x D), X01 and X02 are interchangable
% X11 -- run 1 of week 1 FC (N x D)
% X12 -- run 2 of week 1 FC (N x D), X11 and X12 are interchangable
% Y --  data vector for prediction target (N x 1)
% W -- initial guess of W (D x PD)
% G1 -- initial guess of G1 (N x PD)
% G2 -- initial guess of G2 (N x PD)
% beta --  initial guess of beta (PD x 1)
% Kclass -- the relative weight of classification terms
% lambda_dim -- hyperparameter for dimension identification (L0-regularization)
% lambda_class -- hyperparameter for classification task (L0-regularization)
% max_iter -- max number of iterations
% tol -- error tolerance (stop sign for optimization)
% *** Note, X's should be normalized so that regularization is fair
% for each features. Y is also better to be normalized **
%
% Outputs:
% M -- mask for dimensions with differential information
% Mbeta -- mask for feature weights of differential dimensions for target
% W -- dimension loadings for differential dimensions
% beta -- predictive feature weights of differential dimensions for target
% Others are for troubleshooting purposes

%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu

% Equivalent Kclass
nSub = length(Y);
Kclass = Kclass/nSub;

% tol = 10^-8;
% Set temperature schedules of sigmoid functions for mask dichotomization
KW = 2^(1/5);%1.5^(1/5)
alphaW = 1/KW; % the first alpha is 1
Kbeta = 1.5^(1/5);%2^(1/5)
alphaBeta = 1/Kbeta; % the first alpha is 1

iIter = 1;

% learning rates
% mu_dim = 0.01;
% mu_pred = 0.01;
% mu_maskB = 0.01;
% mu_mask = 0.01;
% mu_G = 0.01;

sigm = @(x) 1./(1+exp(-x));
dsigm = @(a,m) a./(1+exp(-a*m))-a./((1+exp(-a*m)).^2);
% Initialization of mask matrices
M = zeros(size(W));
Mbeta = zeros(size(beta));

% L: the loss function
nSub = length(Y);
maskD = sigm(alphaW*M);
Wstar = maskD.*W;
Gbar = (G1 + G0)/2;
noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
    norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
    norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
L_SNR = noisePower/signalPower;

G = G1 - G0;
maskbeta = sigm(alphaBeta*Mbeta);
gammaD = maskbeta.*beta;
Lclass = -Y'*G*gammaD;
for iSub = 1:nSub
    Gi = G(iSub,:);
    Lclass = Lclass + log(exp(Gi*gammaD)+1);
end
L_new = L_SNR + Kclass * Lclass + lambda_dim*sum(sum(abs(maskD))) + lambda_class*sum(abs(maskbeta));
L = 2*L_new;% for consistency of first loss decrease check

L_history = L_new;
L_sub = [];
alpha_thresh = 50; % stop sign for adequate dichotomization
iterFlag = 1;
convergeThres = 5;

while alphaBeta < alpha_thresh % the outer loop: temperature increase
    alphaW = alphaW * KW;
    alphaBeta = alphaBeta * Kbeta;
    convergeFlag = 0;
    % reset learning rate
    mu_dim = 0.1;
    mu_pred = 0.1;
    mu_maskB = 0.1;
    mu_mask = 0.1;

    while (convergeFlag <= convergeThres) && iIter < max_iter % the inner loop: gradient descent iteration
        if L_new < tol
            break
        end
        L = L_new;
        if (alphaW > 16) && iterFlag
            max_iter = max_iter*1;
            iterFlag = 0;
        end

        % auxiliary variables to improve efficiency
        maskbeta = sigm(alphaBeta*Mbeta);
        maskD = sigm(alphaW*M);
        gammaD = maskbeta.*beta;
        Wstar = maskD.*W;
        Gbar = (G1 + G0)/2;
        G = G1 - G0;
        noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
            norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
        signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
            norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
        if signalPower == 0
            error('Signal Power is 0!')
        end
        L_SNRstar = noisePower/signalPower;

        Gres_noise = X11'*(X11*Wstar-G1) + X12'*(X12*Wstar-G1) + X01'*(X01*Wstar-G0) + X02'*(X02*Wstar-G0);
        Gres_signal = X11'*(X11*Wstar-Gbar) + X12'*(X12*Wstar-Gbar) + X01'*(X01*Wstar-Gbar) + X02'*(X02*Wstar-Gbar);
        Gkernel_dim = (Gres_noise-L_SNRstar*Gres_signal)/signalPower;
        
        % gradient descents (updates of W,M,beta,Mbeta in Box 3.2) 
        W_new = W - mu_dim*Gkernel_dim.*maskD;
        M_new = M - mu_mask*(Gkernel_dim.*W+lambda_dim).*dsigm(alphaW,M);
        beta_new = beta + mu_pred*Kclass*G'*Y.*maskbeta;
        Mbeta_new = Mbeta + mu_maskB*(Kclass*G'*Y.*beta-lambda_class).*dsigm(alphaBeta,Mbeta);
        for iSub = 1:nSub
            Gi = G(iSub,:);
            pY = exp(Gi*gammaD);
            beta_new = beta_new - mu_pred*Kclass*pY/(pY+1)*Gi'.*maskbeta;
            Mbeta_new = Mbeta_new - mu_maskB*Kclass*pY/(pY+1)*Gi'.*beta.*dsigm(alphaBeta,Mbeta);
        end
        % update G (updates of G1 and G0 in Box 2.2) 
        Wstar_new = W_new.*sigm(alphaW*M_new);
        maskbeta_new = sigm(alphaBeta*Mbeta_new);
        gammaD_new = maskbeta_new.*beta_new;
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
        L_new = L_SNR_new + Kclass * Lclass_new + lambda_dim*sum(sum(abs(sigm(alphaW*M_new)))) + lambda_class*sum(abs(sigm(alphaBeta*Mbeta_new)));

        if L_new > L && iIter > 1
            mu_dim = mu_dim/2;
            mu_pred = mu_pred/2;
            mu_maskB = mu_maskB/2;
            mu_mask = mu_mask/2;
%             disp('learning rate reduced')
        else    
            L_history = [L_history,L_new];
            W = W_new;
            M = M_new;
            Mbeta = Mbeta_new;
            beta = beta_new;
            G1 = G1_new;
            G0 = G0_new;
            iIter = iIter + 1;
        end

        G = G1 - G0;
        Gbar = (G1+G0)/2;
        maskbeta = sigm(alphaBeta*Mbeta);
        gammaD = maskbeta.*beta;
        Lclass = -Y'*G*gammaD;
        for iSub = 1:nSub
            Gi = G(iSub,:);
            Lclass = Lclass + log(exp(Gi*gammaD)+1);
        end
        Lclass = Kclass * Lclass;
        noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
            norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
        signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
            norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
        L_SNR = noisePower/signalPower;
        LR_dim = lambda_dim*sum(sum(abs(sigm(alphaW*M))));
        LR_class = lambda_class*sum(abs(sigm(alphaBeta*Mbeta)));
        L_sub = [L_sub,[L_SNR;Lclass;LR_dim;LR_class]];

        if abs(L-L_new) > tol
            convergeFlag = 0;
        else
            convergeFlag = convergeFlag + 1;
        end
    end
    iIter = 0;
end


end