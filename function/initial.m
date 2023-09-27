
function [opt, modelparameter] =  initialization
    modelparameter.addRate = 0.6; 
    opt.lambda1 = 2^4;  % U��21����
    opt.lambda2 = 2^-2;  % ����ʵ��ground_truth label����
    opt.lambda3 = 2^-4;  % ��������ϵ��  
    opt.lambda4 = 2^2;  % 1������ϵ��  
    
    opt.max_iter = 400;
    opt.rho =2^1;   
    
    opt.minimumLossMargin = 0.0001;  
    opt.mode = 1;  
    
    opt.isBacktracking    = 0;    
    opt.tuneThreshold = 1;  
    opt.eta = 1.1;  
    
   %% Model Parameters    
    modelparameter.init = 0;  
end



