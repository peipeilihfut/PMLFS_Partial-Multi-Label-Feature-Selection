clc;
clear;
addpath(genpath('.'));
tic;
 
Num = 10;Smooth = 1; 

load('mirflickr.mat');
warning off;

%% 处理数据
PL = partial_labels;
num_data  = size(data,1);
randorder = randperm(num_data);
[train_data,train_target,~,~ ] = generateCVSet(data,PL',randorder,1,5);
[~,~,test_data,test_target ] = generateCVSet(data,target',randorder,1,5);
[N, num_feature] = size(train_data); 
[num_label, ~] = size(train_target); 


train_target(train_target==-1) = 0;    
test_target(test_target==0) = -1;    
[opt, modelparameter] =  initial;  
init = modelparameter.init;

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
    
    model = PMLFS_1(train_data, train_target, train_target, opt);
    W = model.U; 
    W(size(W, 1), :) = []; 

index = 0;
[dumb , idx] = sort(sum(W.*W,2),'descend'); 
    
NumOfInterval = 20; 
D = ceil(num_feature/2);
step = ceil(D/NumOfInterval);  
    
iterResult = zeros(15, 20); 
    
for d= 1:step:D
     order = (d-1)/step+1;
     fea = idx(1:d);
        
     [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,fea),train_target',Num,Smooth);
    
     [~,~,~,~,~,~,Outputs,Pre_Labels]=...
            MLKNN_test(train_data(:,fea),train_target',test_data(:,fea),test_target',Num,Prior,PriorN,Cond,CondN);
      fprintf('-- Evaluation\n');
               tmpResult = EvaluationAll(Pre_Labels,Outputs,test_target');
               iterResult(:,order) = iterResult(:,order) + tmpResult;
end
Avg_Result      = zeros(15,2);
Avg_Result(:,1) = mean(iterResult,2);

toc;

