function predict_target = Predict(Outputs,tau)

    predict_target = zeros(size(Outputs));
    num_class = size(Outputs,1);
    for l = 1:num_class 
        predict_target(l,:) = Outputs(l,:) >= tau(1,l);
    end
    %predict_target = predict_target*2-1;
end