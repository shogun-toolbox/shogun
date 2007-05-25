A=rand(100);
A(A<0.7)=0;
A=sparse(A);
sg('set_features', 'TRAIN', A);
B=sg('get_features', 'TRAIN');
