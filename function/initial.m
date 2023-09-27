
function [opt, modelparameter] =  initialization
    modelparameter.addRate = 0.6; 
    opt.lambda1 = 2^4;  % U的21范数
    opt.lambda2 = 2^-2;  % 相似实例ground_truth label相似
    opt.lambda3 = 2^-4;  % 迹范数的系数  
    opt.lambda4 = 2^2;  % 1范数的系数  
    
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



