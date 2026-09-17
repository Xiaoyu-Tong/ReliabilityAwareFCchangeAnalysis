function [W,G0,G1,beta,M,Mbeta,alphaW,alphaBeta,L_history,L_sub] = optimizeRegression_ContinuousSparsification(X01,X02,X11,X12,G0,G1,Y,W,beta,Kpred,lambda_dim,lambda_pred,max_iter,tol)
% 
% *** This function implements the optimization procedure as described in
% Box 2.2 in the Appendix of accompanying manuscript.
% This procedure is obtained from the gradient descents in Box 2.1 and
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
% Kpred -- the relative weight of prediction terms
% lambda_dim -- hyperparameter for dimension identification (L0-regularization)
% lambda_pred -- hyperparameter for prediction task (L0-regularization)
% max_iter -- max number of iterations
% tol -- error tolerance (stop sign for optimization)
% *** Note, X's should be normalized so that regularization is fair
% for each features. Y is also better to be normalized **
%
% Outputs:
% M -- mask for brain dimensions 
% Mbeta -- mask for feature weights of brain dimensions
% W -- dimension loadings for brain dimensions
% beta -- predictive feature weights of brain dimensions
% Others are for troubleshooting purposes
%
%
% ** Note: The learning settings, including learning rates and temperature 
% schedules of sigmoid functions are adjustable, and may need customization
% for use in a different problem. **

%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu


% tol = 10^-8;
% Set temperature schedules of sigmoid functions for mask dichotomization
KW = 2^(1/5);%1.5^(1/5)
alphaW = 1/KW; % the first alpha is 1
Kbeta = 1.5^(1/5);%2^(1/5)
alphaBeta = 1/Kbeta; % the first alpha is 1

iIter = 0;

% learning rates
% mu_dim = 0.01;
% mu_pred = 0.01;
% mu_maskB = 0.01;
% mu_mask = 0.01;

sigm = @(x) 1./(1+exp(-x));
dsigm = @(a,m) a./(1+exp(-a*m))-a./((1+exp(-a*m)).^2);
% Initialization of mask matrices
M = zeros(size(W));
Mbeta = zeros(size(beta));

% L: the loss function
Wstar = sigm(alphaW*M).*W;
Gbar = (G1 + G0)/2;
noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
    norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
    norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;

L_new = noisePower/signalPower + ...
    Kpred * norm(Y-(G1-G0)*(sigm(alphaBeta*Mbeta).*beta),"fro")^2 + ...
    lambda_dim*sum(sum(abs(sigm(alphaW*M)))) + lambda_pred*sum(abs(sigm(alphaBeta*Mbeta)));
L = 2*L_new;% for consistency of first loss decrease check

L_history = L_new;
L_sub = [];
alpha_thresh = 500; % stop sign for adequate dichotomization -- adjustable
iterFlag = 1;
convergeThres = 5;

while alphaBeta < alpha_thresh % the outer loop: temperature increase
    alphaW = alphaW * KW;
    alphaBeta = alphaBeta * Kbeta;
    convergeFlag = 0;
    % reset learning rate
    mu_dim = 0.01;
    mu_pred = 0.01;
    mu_maskB = 0.01;
    mu_mask = 0.01;
    while (convergeFlag <= convergeThres) && iIter < max_iter % the inner loop: gradient descent iteration
        if L_new < tol
            break
        end
        L = L_new;
        if (alphaW > 16) && iterFlag
            max_iter = max_iter*4;
            iterFlag = 0;
        end

        % auxiliary variables to improve efficiency
        maskbeta = sigm(alphaBeta*Mbeta);
        maskD = sigm(alphaW*M);
        gammaD = maskbeta.*beta;
        Wstar = maskD.*W;
        Gbar = (G1 + G0)/2;
        noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
            norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
        signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
            norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
        L_SNR = noisePower/signalPower;
        Gkernel_pred = (G1-G0)'*((G1-G0)*gammaD-Y);
        Gres_noise = X11'*(X11*Wstar-G1) + X12'*(X12*Wstar-G1) + X01'*(X01*Wstar-G0) + X02'*(X02*Wstar-G0);
        Gres_signal = X11'*(X11*Wstar-Gbar) + X12'*(X12*Wstar-Gbar) + X01'*(X01*Wstar-Gbar) + X02'*(X02*Wstar-Gbar);
        Gkernel_dim = (Gres_noise-L_SNR*Gres_signal)/signalPower;
        
        % gradient descents (updates of W,M,beta,Mbeta in Box 2.2 of the Appendix) 
        W_new = W - mu_dim*Gkernel_dim.*maskD;
        M_new = M - mu_mask*(Gkernel_dim.*W+lambda_dim).*dsigm(alphaW,M);
        beta_new = beta - mu_pred*2*Kpred*Gkernel_pred.*maskbeta;
        Mbeta_new = Mbeta - mu_maskB*((2*Kpred*Gkernel_pred.*beta+lambda_pred).*dsigm(alphaBeta,Mbeta));

        % instant optimal values of G (updates of G1 and G0 in Box 2.2 of the Appendix) 
        gammaD_new = sigm(alphaBeta*Mbeta_new).*beta_new;
        Wstar_new = sigm(alphaW*M_new).*W_new;
        Ytilde = (X11+X12-X01-X02)/2*Wstar_new*gammaD_new;

        Gbar = (X11+X12+X01+X02)*Wstar_new/4;
        V_XW = norm(X11*Wstar_new - Gbar,"fro")^2 + norm(X12*Wstar_new - Gbar,"fro")^2 ...
            + norm(X01*Wstar_new - Gbar,"fro")^2 + norm(X02*Wstar_new - Gbar,"fro")^2;
        scaler = 2+2*Kpred*V_XW*(gammaD_new'*gammaD_new);

        G1_new = (X11+X12)*Wstar_new/2 + Kpred*V_XW/scaler*(Y-Ytilde)*gammaD_new';
        G0_new = (X01+X02)*Wstar_new/2 - Kpred*V_XW/scaler*(Y-Ytilde)*gammaD_new';

        % new loss function
        Gbar_new = (G1_new + G0_new)/2;
        noisePower_new = norm(G1_new - X11*Wstar_new,"fro")^2 + norm(G1_new - X12*Wstar_new,"fro")^2 +...
            norm(G0_new - X01*Wstar_new,"fro")^2 + norm(G0_new - X02*Wstar_new,"fro")^2;
        signalPower_new = norm(Gbar_new - X11*Wstar_new,"fro")^2 + norm(Gbar_new - X12*Wstar_new,"fro")^2 +...
            norm(Gbar_new - X01*Wstar_new,"fro")^2 + norm(Gbar_new - X02*Wstar_new,"fro")^2;
        L_SNR_new = noisePower_new/signalPower_new;

        L_new = L_SNR_new + Kpred * norm(Y-(G1_new-G0_new)*(sigm(alphaBeta*Mbeta_new).*beta_new),"fro")^2 + ...
            lambda_dim*sum(sum(abs(sigm(alphaW*M_new)))) + lambda_pred*sum(abs(sigm(alphaBeta*Mbeta_new)));

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
        noisePower = norm(G1 - X11*Wstar,"fro")^2 + norm(G1 - X12*Wstar,"fro")^2 +...
            norm(G0 - X01*Wstar,"fro")^2 + norm(G0 - X02*Wstar,"fro")^2;
        signalPower = norm(Gbar - X11*Wstar,"fro")^2 + norm(Gbar - X12*Wstar,"fro")^2 +...
            norm(Gbar - X01*Wstar,"fro")^2 + norm(Gbar - X02*Wstar,"fro")^2;
        L_SNR = noisePower/signalPower;
        LP = Kpred * norm(Y-(G1-G0)*(sigm(alphaBeta*Mbeta).*beta),"fro")^2;
        LR_dim = lambda_dim*sum(sum(abs(sigm(alphaW*M))));
        LR_pred = lambda_pred*sum(abs(sigm(alphaBeta*Mbeta)));
        L_sub = [L_sub,[L_SNR;LP;LR_dim;LR_pred]];

        if abs(L-L_new) > tol
            convergeFlag = 0;
        else
            convergeFlag = convergeFlag + 1;
        end
    end
    iIter = 0;
end

end