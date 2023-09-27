function pLabels = createPL(train_target)
 
pLabels = [];
S = 1 - pdist2(train_target+eps, train_target+eps, 'cosine');  % 标签相似度
 
N = size(train_target, 2);  % 实例数
Q = size(train_target, 1); % 标签数
Ins_labSort = zeros(N, Q); 
label_idx = zeros(N, Q); 
trueLabNum = zeros(1, N);

for i = 1: N  
   trueLabel = find(train_target(:, i)==1); 
   trueLabel = trueLabel';  
   for j = 1:length(trueLabel)
       Ins_labSort(i, :) = Ins_labSort(i, :)  + S(trueLabel(j), :);  
   end
   
   [Ins_labSort(i,:), label_idx(i,:)] = sort(Ins_labSort(i,:),  'descend');  
   trueLabNum(1, i) = length(trueLabel);   
end
PL50 = zeros(N, Q);
PL100 = zeros(N, Q);
PL150 = zeros(N, Q);
for i = 1: N
    if floor(trueLabNum(1,i)*1.5) < Q
        PL50(i, label_idx(1, 1:floor(trueLabNum(1,i)*1.5))) = 1;
    else
        PL50(i, label_idx(1, :)) = 1;
    end
    if floor(trueLabNum(1,i)*2) < Q
        PL100(i, label_idx(1, 1:floor(trueLabNum(1,i)*2))) = 1;
    else
        PL100(i, label_idx(1, :))= 1;
    end
    if floor(trueLabNum(1,i)*2.5) < Q
        PL150(i, label_idx(1, 1:floor(trueLabNum(1,i)*2.5))) = 1;
    else
        PL150(i, label_idx(1, :)) = 1;
    end
end
  pLabels.PL50 = PL50;
  pLabels.PL100 = PL100;
  pLabels.PL150 = PL150;

end

