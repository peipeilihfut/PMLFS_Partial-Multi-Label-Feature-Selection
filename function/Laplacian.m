function L = Laplacian( X, mode )
if mode == 1
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = 1;

    S = constructW(X, options) ;  
    L = diag(sum(S, 2))- S; 
else 
    para = [];
    para.k = 5;
    para.sigma = 1;
    L = Laplacian_GK(X', para);
    
end
end

