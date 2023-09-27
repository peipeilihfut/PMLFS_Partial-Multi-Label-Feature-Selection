addpath(genpath('.'));
clear; clc; 
tic;
name = 'emotions';
load([name '.mat']);
observe = 1;   

 [opt, modelparameter] =  initial;  
 init = modelparameter.init;   
 addRate = modelparameter.addRate; 

 addRate = 0;   
%[optmParameter, modelparameter] =  initialization; 
%model_LSML.optmParameter = optmParameter;
%model_LSML.modelparameter = modelparameter;
%model_LSML.tuneThreshold = 1; 
lambda1_range = 2.^[-6:1:-3];    
lambda2_range = 2.^[-6:1:-3];    
lambda3_range = 2.^[2:2:8];    
lambda4_range = 2.^[2:2:6];        
rho_range = 2.^[1];
totalNum = length(lambda1_range)*length(lambda2_range)*length(lambda3_range)*length(lambda4_range)*length(rho_range);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 if  init == 1
        test_data      = double (test_data);
        train_data      = double (train_data);
        temp_data1 = test_data + eps;
        temp_data2 = train_data + eps;
        temp_data1 = temp_data1./repmat(sqrt(sum(temp_data1.^2,2)),1,size(temp_data1,2));
        if sum(sum(isnan(temp_data1)))>0
            temp_data1 = test_data+eps;
            temp_data1 = temp_data1./repmat(sqrt(sum(temp_data1.^2,2)),1,size(temp_data1,2));
        end
        temp_data2 = temp_data2./repmat(sqrt(sum(temp_data2.^2,2)),1,size(temp_data2,2));
        if sum(sum(isnan(temp_data2)))>0
            temp_data2 = train_data+eps;
            temp_data2 = temp_data2./repmat(sqrt(sum(temp_data2.^2,2)),1,size(temp_data2,2));
        end
 end
 
  
  
  train_target(train_target==-1)=0;  
  PL = getPartialLabel(train_target', addRate, 0);   
  
  [N, num_feature] = size(train_data);  
  [num_label, ~] = size(train_target);  

index = 0;
CurrentResult = zeros(15,2);    
BestResult = zeros(15,2);       
BestIterResult = zeros(15, 20); 

 NumOfInterval = 20;  
 D = ceil(num_feature/2);
 step = ceil(D/NumOfInterval);  
iter = 0;
for a = 1:length(lambda1_range)
   for b = 1:length(lambda2_range)
      for c=1:length(lambda3_range)
          for e=1:length(lambda4_range)
         for d=1:length(rho_range)
             iter = iter+1;
             fprintf('第%d次穷举，共%d次\n',iter,totalNum);
             opt.lambda1 = lambda1_range(a);
             opt.lambda2 = lambda2_range(b);
             opt.lambda3 = lambda3_range(c);
             opt.lambda4 = lambda4_range(e);
             opt.rho = rho_range(d);
             fprintf('lamnda1-4=%.4f, %.4f,%.4f,%d,%.4f\n',lambda1_range(a),lambda2_range(b),lambda3_range(c),lambda4_range, rho_range(d));
             iterResult  = zeros(15,20); 
          
            %% Training
             model = PMLFS_1(train_data, PL, train_target', opt);
             W = model.U;  
             W(size(W, 1), :) = [];  
             [dumb,index] = sort(sum(W.*W,2),'descend');  
                            
            %% Prediction and evaluation
             Num = 10;Smooth = 1;   
            PL(PL==0) = -1;   
                 
            for d = 1:step:D   
               order = (d-1)/step+1;
               f=index(1:d);   
               [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f),PL',Num,Smooth);
               [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
                MLKNN_test(train_data(:,f), PL', test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);
               fprintf('-- Evaluation\n');
               tmpResult = EvaluationAll(Pre_Labels,Outputs,test_target); 
               iterResult(:,order) = iterResult(:,order) + tmpResult; 
             end
                  
             CurrentResult(:,1) = mean(iterResult,2); 
             CurrentResult(:,2) = std(iterResult,1,2);
              
             if(observe == 1)   
                 currAP = CurrentResult(12,1);
                 bestAP = BestResult(12, 1);
                 if(currAP>bestAP) 
                     BestResult = CurrentResult;
                     BestIterResult  = iterResult;
                     BestParameter = opt;
                 end
             else    
                SumCurr= CurrentResult(1,1) + CurrentResult(12,1)  + CurrentResult(13,1) + CurrentResult(14,1) + CurrentResult(15,1);
                SumBest = BestResult(1,1) + BestResult(12,1) + BestResult(13,1) + BestResult(14,1)+ BestResult(15,1);
                if(SumBest == 0)   
                    BestResult = CurrentResult;
                    BestParameter = opt;
                    BestIterResult  = iterResult;
    
                lseif(SumCurr < SumBest)   
                    BestResult = CurrentResult;
                    fprintf('SumBest=%.3f\n',SumCurr);
                    BestParameter = opt;  
                    BestIterResult  = iterResult;    
                end
             end
         end 
          end
      end   
   end
end

toc
