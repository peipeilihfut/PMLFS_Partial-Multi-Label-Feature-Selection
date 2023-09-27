function model = PMLFS(train_data, pLabels, train_target, opt)
% This function is the training phase of the PMLFS algorithm. 
%
%    Syntax
%
%      model =  PMLFS(train_data, pLabels, train_target, opt)

%    Description
%
%       PARMFS takes,
%           train_data                - A NxD array, the instance of the i-th PML example is stored in train_data(i,:)
%           pLabels                   - A NxQ array, if the jth class label is one of the candidate labels for the i-th PML example, then train_target(i,j) equals 1, otherwise train_target(i,j) equals 0
%           train_target              - A NxQ arrat, ground-truth label,²»¿ÉÓÃ   
%           opt.lambda1               - The balancing parameter 
%           opt.lambda2               - The balancing parameter 
%           opt.lambda3               - The balancing parameter 
%           opt.lambda4               - The balancing parameter 
%           opt.max_iter              - The maximum iterations
%       
%       and returns,
%           model                     - The learned model

lambda1 = opt.lambda1;
lambda2 = opt.lambda2;
lambda3 = opt.lambda3;
lambda4 = opt.lambda4;
max_iter = opt.max_iter;

model = [];

[num_train,dim]=size(train_data);   
[~,num_label]=size(pLabels);  

%% Training
fea_matrix = [train_data, ones(num_train,1)];  
V = zeros(dim+1,num_label);  
U = zeros(dim+1,num_label);
Y = zeros(dim+1,num_label);
mu = 1e-4;    
rho = 1.1;    
epsilon = 1e-5;  

XTY = fea_matrix'*pLabels;  
XTX = fea_matrix'*fea_matrix;  
W = (XTX+eye(dim+1))\(XTY+mu*U+mu*V+Y);
%D = diag(1/(sqrt(sum(W.*W, 2)) + epsilon)); 
 
for t = 1: max_iter 
    D = diag(1/(sqrt(sum(W.*W, 2)) + epsilon)); 
    W = (XTX+lambda1*D)\(XTY+mu*U+mu*V+Y);  
    
    Uk=U;
    Vk=V;
    [M,S,Nhat] = svd(W-V-Y/mu,'econ'); 
    sp = diag(S);   
    svp = length(find(sp>lambda2/mu));   
    if svp>=1
        sp = sp(1:svp)-lambda2/mu;
    else
        svp=1;
        sp=0;
    end
    Uhat =  M(:,1:svp)*diag(sp)*Nhat(:,1:svp)' ;   
    U = Uhat;
    
    % L1 norm
    Vraw = W-U-Y/mu;
    Vhat = max(Vraw - lambda3/mu,0)+min(Vraw+lambda3/mu,0);  
    V = Vhat;
  
    convg2 = false;   
    stopCriterion2 = mu*norm(U-Uk,'fro')/norm(W,'fro');  
    if stopCriterion2<1e-5
        convg2=true;
    end
    convg1 = false;
    tmp = W-U-V;
    stopCriterion1 = norm(tmp,'fro')/norm(W,'fro');  
    if stopCriterion1<1e-7
        convg1 = true;
    end
    
    if convg2   
        mu = min(rho*mu,1e10);
    end
    Y = Y+mu*(U+V-W);   
    
    if convg1 && convg2  
        break;
    end
end

model.W = W;  
model.U = U;
model.V = V;
end

