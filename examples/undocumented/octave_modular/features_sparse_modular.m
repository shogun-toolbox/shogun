modshogun
rand('state',0);

disp('Sparse Features')

A=rand(3,5);
A(A<0.7)=0;
full(A)

% sparse representation X of dense matrix A
X = sparse(A)

% create sparse shogun features from dense matrix A
a=SparseRealFeatures(A)
a_out = a.get_full_feature_matrix()

% create sparse shogun features from sparse matrix X
a.set_sparse_feature_matrix(X)
a_out=a.get_full_feature_matrix()

% create sparse shogun features from sparse matrix X
a=SparseRealFeatures(X)
a_out=a.get_full_feature_matrix()

z=a.get_sparse_feature_matrix()

