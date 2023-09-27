function [ Y ,realpercent] = getPartialLabel( Y, percent, bQuiet )

obsTarget_index = zeros(size(Y));

totoalNum = sum(sum(Y ~=0));
totoalAddNum = 0;  
[N,~] = size(Y);
realpercent = 0;  
maxIteration = 50;
factor = 2;
count=0;
if percent > 0
    while realpercent < percent
        if maxIteration == 0
            factor = 1;  
            maxIteration = 10;
            if count==1   
                break;
            end
            count = count+1;
        else
            maxIteration = maxIteration - 1;
        end
        for i=1:N 
            index = find(Y(i,:)~=1); 
            if length(index) >= factor  
                addNum = round(rand*(length(index))); 
                totoalAddNum = totoalAddNum + addNum;
                realpercent = totoalAddNum/totoalNum;
            
                if addNum > 0
                    index = index(randperm(length(index))); 
                    Y(i,index(1:addNum)) = 1; 
                    obsTarget_index(i,index(1:addNum))= 1;
                end
            
                if realpercent >= percent
                    %满足缺失要求，跳出循环
                    break;
                end
            end
        end
    end
end

if bQuiet == 0
    fprintf('\n  Totoal Number of Feature Entities : %d\n ',totoalNum);  
    fprintf('Number of Deleted Feature Entities : %d\n ',totoalAddNum);  
    fprintf('        Given percent/Real percent : %.2f / %.2f\n', percent,totoalAddNum/totoalNum);  
end

end
