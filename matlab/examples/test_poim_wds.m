addpath( '../../src/' );


% === const
acgt = 'ACGT';
% --- data
len = 11;
num_train = 2000;
num_test  = 1000;
% --- POIM
max_order = 8;
% --- kernel
order = 6;
shift = 0;
max_mismatch = 0;
% --- SVM
C = 1 / len^2;
cache = 10;


% === init
shifts = sprintf( '%i ', shift*ones(1,len) );
num = num_train + num_test;
rand( 'seed', 17 );
rand( 'state', 1 );


% === generate toy data
% --- generate all data
dat = acgt( floor( 4 * rand(len,num) ) + 1 );
lab = ( (-1) .^ (1:num) );
for( i = find(lab==+1) )
  dat( 3:5, i ) = 'AAA';
end;
% --- split to training and test
traindat = dat( :, 1:num_train );
trainlab = lab(    1:num_train );
testdat = dat( :, num_train + (1:num_test) );
testlab = lab(    num_train + (1:num_test) );


% === train SVM
sg( 'send_command', 'loglevel INFO' );
sg( 'send_command', 'use_linadd 1' );
sg( 'send_command', 'use_batch_computation 1' );
sg( 'set_features', 'TRAIN', traindat, 'DNA' );
sg( 'set_labels', 'TRAIN', trainlab );
sg( 'send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS2_NONORM CHAR 10 %i %i %i %s', order, max_mismatch, len, shifts ) );
sg( 'send_command', 'init_kernel TRAIN' );
sg( 'send_command', 'new_svm LIGHT' );
sg( 'send_command', sprintf('c %f',C) );
sg( 'send_command', 'svm_train' );


% === test
if( 0 )
  t1 = 'CCCCACCCCCC';
  t2 = 'CCCCCCCCCCC';  
  T = [ t1 ; t2 ]' 
  sg( 'set_features', 'TEST', T, 'DNA' );
  sg( 'set_labels', 'TEST', ones(1,size(T,2)) );
  sg( 'send_command', 'init_kernel TEST' );
  out = sg( 'svm_classify' )
end;


% % === evaluate SVM on test data
% sg( 'set_features', 'TEST', testdat, 'DNA' );
% sg( 'set_labels', 'TEST', testlab );
% sg( 'send_command', 'init_kernel TEST' );
% out = sg( 'svm_classify' );
% fprintf( 'accuracy: %f\n', mean(sign(out)==testlab) );


% === compute POIMs
Q = sg( 'compute_poim_wd', max_order );
x = {};
X = zeros( max_order, len );
l = 0;
for( k = 1:max_order )
  L = l + 4^k*len;
  q = Q((l+1):L);
  q = reshape( q, [4^k,len] );
  q = q - repmat( mean(q,1), 4^k, 1 );
  q( :, (len-k+2):len ) = 0;
  x{k} = q;
  l = L;
  X(k,:) = max( abs(x{k}), [], 1 );
  %X(k,:) = var( x{k} );
end;
%save( 'S.mat', 'x', 'X' );


% === output
figure;
for( i = 1:4 )
  subplot( 2, 2, i );
  imagesc( x{i} );
end;
figure;
imagesc( X );


% === predict all possible sequences
N = 4^len;
T = repmat( ' ', len, N );
t = (1:N) - 1;
for( i = len:-1:1 )
  T(i,:) = acgt( mod(t,4)+1 );
  t = floor( t / 4 );
end;
sg( 'set_features', 'TEST', T, 'DNA' );
sg( 'set_labels', 'TEST', ones(1,N) );
sg( 'send_command', 'init_kernel TEST' );
out = sg( 'svm_classify' );


% === compute true POIMs
poims = {};
meanOut = mean( out );
%for( k = 1:max_order )
for( k = 1:3 )
  m = 4^k;
  poim = zeros( m, len );
  t = (1:N) - 1;
  for( i = (len-k+1):-1:1 )
    y = mod( t, m ) + 1;
    for( z = 1:m )
      poim(z,i) = mean( out(y==z) );
    end;
    t = floor( t / 4 );
  end;
  poim = poim - meanOut;
  poim( :, (len-k+2):len ) = 0;
  poims{k} = poim;
end;


% === compare
for( k = 1:length(poims) )
  if( 0 )
    figure;
    imagesc( x{k} );
    title( sprintf( '%d (shogun)', k ) );
    figure;
    imagesc( poims{k} );
    title( sprintf( '%d (truth)', k ) );
  end;
  fprintf( 'order %d: norm diff = %.2e \n', k, norm(poims{k}-x{k}) );
end;

