% 每次运行会生成新的PML数据集，保留数据集和生成结果（excel表格中）的参数

    clc; clear; 
    tic;  % 计时开始
    addpath(genpath('.\'))
    
    name = 'medical';
    load([name '.mat']);

    [opt, modelparameter] =  initial;  
    init = modelparameter.init;
    addRate = modelparameter.addRate; 

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
    % pLabels = createPL(train_target);   

    [N, num_feature] = size(train_data); 
    [num_label, ~] = size(train_target); 

    model = PMLFS_1(train_data, PL, train_target', opt);

    W = model.U; 
    V = model.V;

    W(size(W, 1), :) = []; 

    [dumb,index] = sort(sum(W.*W,2),'descend'); 
    Num = 10;Smooth = 1;  
    PL(PL==0) = -1;  

    NumOfInterval = 20; 
    D = ceil(num_feature/2);
    step = ceil(D/NumOfInterval);  

    iterResult = zeros(15, 20); 

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
    Avg_Result      = zeros(15,2);
    Avg_Result(:,1) = mean(iterResult,2);
    Avg_Result(:,2) = std(iterResult,1,2);  
        
    toc;
   

