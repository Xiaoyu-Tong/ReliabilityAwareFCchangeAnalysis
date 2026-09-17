function [W,G0,G1,beta] = optimizeClassification_Initialization(X01,X02,X11,X12,Y,Kclass,lambda_dim,lambda_class)
% 
% *** This function implements the optimization procedure as described in
% Box 3.4 in the Appendix of accompanying manuscript.
%
% Inputs:
% X01 -- run 1 of baseline FC (N x D)
% X02 -- run 2 of baseline FC (N x D), X01 and X02 are interchangable
% X11 -- run 1 of week 1 FC (N x D)
% X12 -- run 2 of week 1 FC (N x D), X11 and X12 are interchangable
% Y --  data vector for prediction target (N x 1)
% Kclass -- the relative weight of classification terms
% lambda_dim -- hyperparameter for dimension identification (L0-regularization)
% lambda_class -- hyperparameter for classification task (L0-regularization)
% *** Note, X's should be normalized so that regularization is fair
% for each features. Y is also better to be normalized **
%
% Outputs:
% W -- initial guess for W (brain dimension loadings)
% G0 -- initial guess for G0 (brain dimension score @ t0)
% G1 -- initial guess for G1 (brain dimension score @ t1)
% beta -- initial guess for beta (prediction weights)
%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu

% Initiate W as the dimension where pre-treatment and post-initiation
% measures show the highest correspondance. (Steps 1 and 2 in Box 3.4)
[A0,B0] = canoncorr(X01,X02);
W0 = (A0+B0)/2;
[A1,B1] = canoncorr(X11,X12);
W1 = (A1+B1)/2;
W = (W0+W1)/2;
W = W/norm(W,'fro');

% (Step 3 in Box 3.4)
G0 = (X01+X02)*W/2;
G1 = (X11+X12)*W/2;
G = G1 - G0;
G = G/lambda_class;

% If a simple regression model can be found, use the corresponding weights
% as the initial guess of beta. In the rare case where no such model
% exists, initiate beta as a zero matrix. (Step 4 in Box 3.4)
[Mdl,FitInfo] = fitclinear(G,Y,'Lambda',lambda_class,'Learner','logistic','Regularization','lasso');
beta = Mdl.Beta;

% (Step 5 in Box 3.4)
G0 = (X01+X02)*W/2;
G1 = (X11+X12)*W/2;

end