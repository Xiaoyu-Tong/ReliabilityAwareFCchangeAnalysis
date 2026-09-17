function w = CVQ(X_fg,X_bg,X_noise,rho)
% Conduct the contrastive variance quotient (CVQ) analysis, which
% identifies the dimension that maximizes the ratio between its variances 
% in foreground and background data. In other words, the CVQ analysis 
% extracts the most "foreground-specific" dimension. 
%
% In the application scenario with signal change-based features (e.g., FC
% changes are input features), the CVQ analysis will also minimizes the
% dimension's variance in test-retest difference. This design is to ensure
% that the identified dimension indeed reflects foreground-specific signals
% instead of some spurious noise. For applications where test-retest
% variability is irrelavant, setting the importance of test-retest
% reliability (rho) to zero can remove this term. 
%
% For example use, please see demoMain_CVQ.m
%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu
%
% -------------------------------------------------------------------------
% Inputs:
% X_fg: The foreground data (N_fg x D)
% X_bg: The background data (N_bg x D)
% X_noise: The test-retest difference (i.e., noise data) (N_noise x D)
% rho: The importance of test-retest reliability (relative to the
% background variance). Set to zero if this term is irrelavant.
%
%
% Outputs:
% w: loading of the dimension where the ratio between foreground and
% background variances are maximized.
%
% Note: The dimension scores for foreground and background data are 
% X_fg * w and X_bg * w, respectively.
%

N_fg = size(X_fg,1);
N_bg = size(X_bg,1);
N_noise = size(X_noise,1);
% The foreground, background, and noise data may have different sample
% sizes, but their feature dimensionality must be the same.

% Checkpoint: the matrix in denominator must be Hermitian positive-definite
A = (X_bg'*X_bg)/N_bg + rho*(X_noise'*X_noise)/N_noise; % the matrix in denominator
eigs = eig(A);
eigs = sort(eigs,'ascend');
if eigs(1)<=0
    error('The background data matrix must be full rank!')
end

% Find the dimension where the ratio between foreground and background
% variances are maximized. The CVQ is essentially a generalized Raleigh
% quotient, whose analytical solution can be found in the following way:

% 1. Solve the Cholesky decomposition of background (including noise data)
C = chol(A,"lower");
% 2. Define the generalized Raleigh quotient problem
D = (X_fg'*X_fg)/N_fg;
D = C \ D / C';
% 3. Solve the equivalent Raleigh quotient problem
[rEigVecs,eigVals] = eig(D);
[eigVals,idd] = sort(diag(eigVals),'descend');
rEigVecs = rEigVecs(:,idd);
u = rEigVecs(:,1);
w = C' \ u;


end

