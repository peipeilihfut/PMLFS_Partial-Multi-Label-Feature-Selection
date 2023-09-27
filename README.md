# PMLFS: Partial Multi-Label Feature Selection
This work has been published in International Joint Conference on Neural Networks:
J. Wang, P. Li and K. Yu, "Partial Multi-Label Feature Selection," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-9, doi: 10.1109/IJCNN55064.2022.9892133.
## Abstract
Multi-label feature selection is an effective approach to alleviate the high dimensionality of multi-label learning tasks. Most of the existing multi-label feature selection methods are based on the assumption that the relevant labels of each training sample are precisely annotated. However, in real-world applications, this assumption is not always held, each instance may be labeled with a set of candidate labels that contains all the ground-truth labels adulterated with noisy labels, which is called partial multi-label problem. Previous multi-label feature selection methods can not select the optimal feature set in the presence of partial multi-label. Therefore, to tackle this problem we propose a novel partial multi-label feature selection method, called PMLFS. To be specific, ground-truth labels and noisy labels are firstly distinguished in terms of label correlations and the sparsity of noisy labels. Secondly, manifold regularization is incorporated to explore the local structure. Based on the above, we design an objective framework involving $l_{2,1}$-norm regularization to achieve partial multi-label feature selection. Finally, extensive experiments on synthetic and real-world partial multi-label data sets demonstrate that the proposed method outperforms the state-of-the-art multi-label feature selection methods.
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9892133&isnumber=9889787
