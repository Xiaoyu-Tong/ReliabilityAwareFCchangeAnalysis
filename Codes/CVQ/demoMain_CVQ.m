clear

% This script demonstrates how to conduct the CVQ analysis in a
% cross-validation setting using foreground and background metrics at two
% time points, each with test and retest data acquisition. 

% In this demo, the two time points represent the pre-treatment (t0) and
% post-initiation (t1) recordings of functional connectivity (FC). The data
% for the sertraline arm is foreground, while the data for the placebo arm
% is the backgroud. The noise data is defined as the test-retest
% differences for each data point. Data are replaced with random matrices.
%
% by Xiauyu Tong, Stanford, Last updated 2026-3
% tongxy@stanford.edu

% -------------------------------------------------------------------------
% Compile data (replace with real data in actual use)
n_SER = 123;
n_PLA = 125; % number of subjects in SER and PLA arms
n_total = n_SER + n_PLA;
nFe = 9045;

X01 = randn(n_total,nFe); % test recording @ t0
X02 = randn(n_total,nFe); % retest recording @ t0
X11 = randn(n_total,nFe); % test recording @ t1
X12 = randn(n_total,nFe); % retest recording @ t1
% 9045 is the feature dimensionality of FC with nROI = 135

% X = [X01;X02;X11;X12];
% Each column in X should correspond to the data for the same subject.
% 
subName = cellstr(num2str((1:n_total)')); % replace with actual subject IDs
idSER = false(n_total,1); idSER(randsample(n_total,n_SER)) = 1; % replace with the actual indices of foreground data

subName_PLA = subName(~idSER);
subName_SER = subName(idSER);

% Cross-validation
rho = 0.3; % set hyperparameter

rng('default')
nRound = 5;
nFold = 20; % set number of rounds and folds

% variables for validation result
dimScore_SER = zeros(length(subName_SER),nRound);
dimScore_PLA = zeros(length(subName_PLA),nRound);
CVQ_all = zeros(nRound,1);

for iRound = 1:nRound
    cvidx_PLA=crossvalind('Kfold',subName_PLA,nFold);
    cvidx_SER=crossvalind('Kfold',subName_SER,nFold);
    for iFold = 1:nFold
        % partition subjects
        subID_test_PLA=subName_PLA(cvidx_PLA==iFold); subID_train_PLA=subName_PLA(cvidx_PLA~=iFold);
        subID_test_SER=subName_SER(cvidx_SER==iFold); subID_train_SER=subName_SER(cvidx_SER~=iFold);
        idx_test = find(contains(subName,[subID_test_PLA;subID_test_SER]));
        idx_train = find(contains(subName,[subID_train_PLA;subID_train_SER]));
        
        % partition data
        X01_train = X01(idx_train,:);
        X02_train = X02(idx_train,:);
        X11_train = X11(idx_train,:);
        X12_train = X12(idx_train,:);
        idSER_train = idSER(idx_train);
        X01_test = X01(idx_test,:);
        X02_test = X02(idx_test,:);
        X11_test = X11(idx_test,:);
        X12_test = X12(idx_test,:);
        idSER_test = idSER(idx_test);

        % Normalization
        % To keep change-based information, data from different time points are normalized together.
        X_train = [X01_train;X02_train;X11_train;X12_train];
        X_test = [X01_test;X02_test;X11_test;X12_test];

        X_test = X_test-repmat(mean(X_train),size(X_test,1),1);
        X_test = X_test./repmat(std(X_train),size(X_test,1),1);
        X_train = X_train-repmat(mean(X_train),size(X_train,1),1);
        X_train = X_train./repmat(std(X_train),size(X_train,1),1);

        % re-compile normalized data
        nSub_train = size(X_train,1)/4;
        X01_train = X_train(1:nSub_train,:);
        X02_train = X_train(nSub_train+1:2*nSub_train,:);
        X11_train = X_train(2*nSub_train+1:3*nSub_train,:);
        X12_train = X_train(3*nSub_train+1:4*nSub_train,:);

        nSub_test = size(X_test,1)/4;
        X01_test = X_test(1:nSub_test,:);
        X02_test = X_test(nSub_test+1:2*nSub_test,:);
        X11_test = X_test(2*nSub_test+1:3*nSub_test,:);
        X12_test = X_test(3*nSub_test+1:4*nSub_test,:);

        X01_train_PLA = X01_train(~idSER_train,:);
        X01_train_SER = X01_train(idSER_train,:);
        X02_train_PLA = X02_train(~idSER_train,:);
        X02_train_SER = X02_train(idSER_train,:);
        X11_train_PLA = X11_train(~idSER_train,:);
        X11_train_SER = X11_train(idSER_train,:);
        X12_train_PLA = X12_train(~idSER_train,:);
        X12_train_SER = X12_train(idSER_train,:);

        % FC change data (calculate foreground and background for training set)
        X_bg_train = (X11_train_PLA + X12_train_PLA - X01_train_PLA - X02_train_PLA)/2;
        X_fg_train = (X11_train_SER + X12_train_SER - X01_train_SER - X02_train_SER)/2;
        X_test = (X11_test + X12_test - X01_test - X02_test)/2;
        
        % noise data (test-retest difference)
        XD1_train = (X11_train-X12_train);
        XD0_train = (X01_train-X02_train);

        % Feature selection (optional)
        % Feature selection is necessary if the number of subjects is low
        % relative to feature dimensionality. Essentially, the
        % dimensionality of selected features needs to ensure that the
        % denominator of CVQ has full rank.
        [~,pF] = vartest2(X_bg_train,X_fg_train,'tail','left'); % one-tailed F-test
        pThresh = 0.01; % empirically corresponds to a feature number approximately be the half of subject number (of each group)
        idx = pF < pThresh;
        % optional:
        if sum(idx) >= size(XD0_train,1)/2  % a hard coding procedure to limit the number of selected features, which may be used to enhance stability
            [~,idx] = sort(pF,'ascend');
            idx = idx(1:round(size(XD0_train,1)/2));
        end
        
        % compile selected features
        X_fg_train = X_fg_train(:,idx);
        X_bg_train = X_bg_train(:,idx);
        XD1_train = XD1_train(:,idx);
        XD0_train = XD0_train(:,idx);
        X_noise_train = [XD0_train;XD1_train];

        X_test = X_test(:,idx);
        X01_test = X01_test(:,idx);
        X02_test = X02_test(:,idx);
        X11_test = X11_test(:,idx);
        X12_test = X12_test(:,idx);

        % conduct the CVQ analysis on training data
        w = CVQ(X_fg_train,X_bg_train,X_noise_train,rho);

        % test on the validation fold
        X01_test_PLA = X01_test(~idSER_test,:);
        X01_test_SER = X01_test(idSER_test,:);
        X02_test_PLA = X02_test(~idSER_test,:);
        X02_test_SER = X02_test(idSER_test,:);
        X11_test_PLA = X11_test(~idSER_test,:);
        X11_test_SER = X11_test(idSER_test,:);
        X12_test_PLA = X12_test(~idSER_test,:);
        X12_test_SER = X12_test(idSER_test,:);
        dimScore_PLA(cvidx_PLA==iFold,iRound) = (X11_test_PLA+X12_test_PLA-X01_test_PLA-X02_test_PLA)/2 * w;
        dimScore_SER(cvidx_SER==iFold,iRound) = (X11_test_SER+X12_test_SER-X01_test_SER-X02_test_SER)/2 * w;
    end
    CVQ_all(iRound) = var(dimScore_SER(:,iRound))/var(dimScore_PLA(:,iRound));
end

% The CVQ values should be greater than their critical value (derived from
% F-test) to validate the statistical significance of the
% foreground-specific dimension.





