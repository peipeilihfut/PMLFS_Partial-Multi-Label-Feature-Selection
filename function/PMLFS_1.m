function model = PMLFS_1(X, Y, train_target, opt)
% This function is the training phase of the PMLFS algorithm. 
%
%    Syntax
%
%      model =  PMLFS(train_data, pLabels, train_target, opt)

%    Description
%
%       PARMFS takes,
%           X                   - A NxD array, the instance of the i-th PML example is stored in train_data(i,:)
%           Y                   - A NxQ array, if the jth class label is one of the candidate labels for the i-th PML example, then train_target(i,j) equals 1, otherwise train_target(i,j) equals 0
%           train_target              - A NxQ arrat, ground-truth label,不可用   
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
rho = opt.rho;
eta = opt.eta;
miniLossMargin = opt.minimumLossMargin;
tuneThreshold = opt.tuneThreshold;

isBacktracking = opt.isBacktracking; 
mode = opt.mode; 

 L = Laplacian(X , mode);
%L = Laplacian([X Y], mode);   

model = [];

[num_sample,dim]=size(X);   
[~,num_label]=size(Y);   

%% Training
fea_matrix = [X, ones(num_sample,1)];  

XTY = fea_matrix'*Y;  
XTX = fea_matrix'*fea_matrix;  
XTLX = fea_matrix'*L*fea_matrix;
YTY = Y'*Y; 
% D = diag(1/(sqrt(sum(W.*W, 2)) + eps)); 
%% initialization
 U = (XTX+rho*eye(dim+1))\XTY;  
 V = (XTX+rho*eye(dim+1))\(XTY-XTX*U); 
 U_1 = U; U_k = U;
 V_1 = V;
 iter = 1; oldloss = 0;
 bk = 1; bk_1 = 1;
 
 Lip1 = 6*norm(XTX)^2 + 4*norm(lambda2*fea_matrix'*(L+L')*fea_matrix)^2;
 Lip = sqrt(Lip1);
while iter <= max_iter
    if isBacktracking == 0
        DU = diag(1./(sqrt(sum(U.*U, 2))+ eps));  
       % if iter == 1
       %     DU = diag(1./(sqrt(sum(U.*U, 2))+ eps));  
       % else
       %     DU= diag(1./(sqrt(sum(U.*U, 2))+ eps))- diag(1./(sqrt(sum(U_1.*U_1, 2))+ eps));
       % end
        Lip2 = norm(lambda1*DU)^2;
        Lip = sqrt(Lip1+4*Lip2);
    else
         F_v = calculateF(U, XTX, XTY, YTY,XTLX, V,lambda1, lambda2);
         QL_v = calculateQ(U, XTX, XTY, YTY,XTLX, V, lambda1,lambda2, Lip,U_k);
         while F_v > QL_v
            Lip =  eta*Lip;
            QL_v = calculateQ(U, XTX, XTY, YTY,XTLX, V, lambda1,lambda2, Lip,U_k);
         end
    end
    
     %% update U  
       U_k  = U + (bk_1 - 1)/bk * (U - U_1);
       Gu_k = U_k - 1/Lip * gradientOfU(XTX,XTY,U,V,XTLX,lambda1,lambda2);
      
       [M,S,Uhat] = svd(Gu_k,'econ');  
        sp = diag(S);   
        svp = length(find(sp>lambda3/Lip));   
        if svp>=1
           sp = sp(1:svp)-lambda3/Lip;
        else
           svp=1;
           sp=0;
        end
        Uhat =  M(:,1:svp)*diag(sp)*Uhat(:,1:svp)' ;   
        U = Uhat;
         U_1  = U;
         
      %% update V
       V_k  = V + (bk_1 - 1)/bk * (V - V_1);
       Gv_x_k = V_k - 1/Lip * gradientOfV(XTX,XTY,U,V);
      
       V   = softthres(Gv_x_k,lambda4/Lip);
        V_1  = V;
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       
       %% Loss
       LS = fea_matrix*(U+V) - Y;
       DiscriminantLoss = trace(LS'* LS);
       U21 = sum(sqrt(sum(U.*U,2)+eps));
       CorrelationLoss  = trace(U'*XTLX*U);
       traceOfU    = trace(sqrt(U'*U));
       sparesV    = sum(sum(V~=0));
       
       totalloss = DiscriminantLoss + lambda1*U21 + lambda2*CorrelationLoss + lambda3*traceOfU+lambda4*sparesV;
       loss(iter,1) = totalloss;
       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       iter=iter+1;
 
end

model.XU = getTrueLabel(fea_matrix, U, Y, tuneThreshold);
model.W = U + V;  
model.U = U;
model.V = V;
end

%% 根据XU的值估计ground-truth label 
function XU = getTrueLabel(fea_matrix, U, Y, tuneThreshold)
    Outputs = (fea_matrix*U);
    if tuneThreshold == 1   
        fscore                 = (fea_matrix*U);
        [ tau,  currentResult] = TuneThreshold( fscore, Y, 1, 1);
        XU             = Predict(Outputs,tau);
    else
        XU = double(Outputs>=0.5);
    end

end

%% 软阈值操作
function W = softthres(W_t,lambda)
    W = max(W_t - lambda,0) - max(-W_t - lambda,0);  
end

function gradient = gradientOfU(XTX,XTY,U,V,XTLX,lambda1,lambda2)
    % L = diag(sum(C,2)) - C;
    % D_1 = diag(1/(sqrt(sum((U+V).*(U+V), 2)) + eps)); 
    D = diag(1./(sqrt(sum(U.*U, 2))+ eps));
    gradient = XTX*U + XTX*V - XTY + lambda1*D*U + 2*lambda2*XTLX*U;
end

function gradient = gradientOfV(XTX,XTY,U,V)
    gradient = XTX*U + XTX*V - XTY;
end

function F_v = calculateF(U, XTX, XTY, YTY,XTLX, V,lambda1, lambda2)
% calculate the value of function F原函数
    F_v = 0;
    F_v = F_v + 0.5*trace(U'*XTX*U + U'*XTX*V - V'*XTY + V'*XTX*U + V'*XTX*V- V'*XTY -(XTY)'*U -(XTY)'*V + YTY);
    F_v = F_v + lambda1*sum(sqrt(sum(U.*U,2)+eps));
    F_v = F_v + lambda2*trace(U'*XTLX*U);
end

function QL_v = calculateQ(U, XTX, XTY, YTY,XTLX, V, lambda1,lambda2, Lip,U_t)
% calculate the value of function Q_L
    QL_v = 0;
    QL_v = QL_v +  calculateF(U, XTX, XTY, YTY,XTLX, V,lambda1, lambda2); 
    QL_v = QL_v + 0.5*Lip*norm(U - U_t,'fro')^2;  
    QL_v = QL_v + trace((U - U_t)'*gradientOfU(XTX,XTY,U,V,XTLX,lambda1,lambda2));
end


