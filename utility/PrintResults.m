
function PrintResults(Result)
fprintf('------------------------------------\n');
fprintf('Evalucation Metric    Mean    Std\n');
fprintf('------------------------------------\n');
fprintf('HammingLoss           %.3f  %.3f\r',Result(1,1),Result(1,2));
fprintf('ExampleBasedAccuracy  %.3f  %.3f\r',Result(2,1),Result(2,2));
fprintf('ExampleBasedPrecision %.3f  %.3f\r',Result(3,1),Result(3,2));
fprintf('ExampleBasedRecall    %.3f  %.3f\r',Result(4,1),Result(4,2));
fprintf('ExampleBasedFmeasure  %.3f  %.3f\r',Result(5,1),Result(5,2));

fprintf('SubsetAccuracy        %.3f  %.3f\r',Result(6,1),Result(6,2));
fprintf('LabelBasedAccuracy    %.3f  %.3f\r',Result(7,1),Result(7,2));
fprintf('LabelBasedPrecision   %.3f  %.3f\r',Result(8,1),Result(8,2));
fprintf('LabelBasedRecall      %.3f  %.3f\r',Result(9,1),Result(9,2));
fprintf('LabelBasedFmeasure    %.3f  %.3f\r',Result(10,1),Result(10,2));

fprintf('MicroF1Measure        %.3f  %.3f\r',Result(11,1),Result(11,2));
fprintf('Average_Precision     %.3f  %.3f\r',Result(12,1),Result(12,2));
fprintf('OneError              %.3f  %.3f\r',Result(13,1),Result(13,2));
fprintf('RankingLoss           %.3f  %.3f\r',Result(14,1),Result(14,2));
fprintf('Coverage              %.3f  %.3f\r',Result(15,1),Result(15,2));
fprintf('AUC                   %.3f  %.3f\r',Result(16,1),Result(16,2));
fprintf('------------------------------------\n');
end