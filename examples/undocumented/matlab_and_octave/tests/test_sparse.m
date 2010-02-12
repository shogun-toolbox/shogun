rand('state',0);
A=rand(3,5);
A(A<0.7)=0;
full(A)
%sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', sparse(A));
B=sg('get_features', 'TRAIN');
full(B)

sg('set_features', 'TRAIN', A);
C=sg('get_features', 'TRAIN');

C
