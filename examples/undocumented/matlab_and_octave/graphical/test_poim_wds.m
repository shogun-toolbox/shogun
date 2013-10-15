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
order = 4;
shift = 0;
max_mismatch = 0;
% --- SVM
C = 1 / len^2;
cache = 10;
% --- DEBUG
% 0: POIM
% 1: substring
% 2: superstring
% 3: left overlap
% 4: right overlap
subScore = 0;

% === init
abcSize = length( acgt );
shifts = int32(shift*ones(1,len));
num = num_train + num_test;
rand( 'seed', 17 );
rand( 'state', 1 );
if( 1 )
  distrib = repmat( 1/abcSize, abcSize, len );
else
  distrib = rand( abcSize, len );
  distrib = distrib ./ repmat( sum(distrib,1), abcSize, 1 );
end;


% === generate toy data
% --- generate all data
dat = acgt( floor( abcSize * rand(len,num) ) + 1 );
lab = ( (-1) .^ (1:num) );
for( i = find(lab==+1) )
  dat( 5:7, i ) = 'AAA';
end;
% --- split to training and test
traindat = dat( :, 1:num_train );
trainlab = lab(    1:num_train );
testdat = dat( :, num_train + (1:num_test) );
testlab = lab(    num_train + (1:num_test) );


% === train SVM
sg( 'loglevel', 'INFO' );
sg( 'use_linadd', true);
sg( 'use_batch_computation', true);
sg( 'set_features', 'TRAIN', traindat, 'DNA' );
sg( 'set_labels', 'TRAIN', trainlab );
sg( 'set_kernel', 'WEIGHTEDDEGREEPOS2_NONORM', 'CHAR', 10, order, max_mismatch, len, shifts );
sg( 'new_classifier', 'SVMLIGHT' );
sg( 'c', C);
sg( 'train_classifier' );


% === test
if( 0 )
  t1 = 'CCCCACCCCCC';
  t2 = 'CCCCCCCCCCC';
  T = [ t1 ; t2 ]'
  sg( 'set_features', 'TEST', T, 'DNA' );
  sg( 'set_labels', 'TEST', ones(1,size(T,2)) );
  out = sg( 'classify' )
end;


% % === evaluate SVM on test data
% sg( 'set_features', 'TEST', testdat, 'DNA' );
% sg( 'set_labels', 'TEST', testlab );
% out = sg( 'classify' );
% fprintf( 'accuracy: %f\n', mean(sign(out)==testlab) );


% === compute POIMs
n = sg( 'get_kernel_optimization', max_order );
if( subScore == 0 )
  Q = sg( 'compute_poim_wd', max_order, distrib );
else
  assert( 0 < subScore & subScore <= 4 );
  t = -( max_order*4 + subScore-1 );
  Q = sg( 'compute_poim_wd', t, distrib );
end;
w = {};
W = zeros( max_order, len );
x = {};
X = zeros( max_order, len );
l = 0;
for( k = 1:max_order )
  L = l + abcSize^k*len;
  w{k} = reshape( n((l+1):L), [abcSize^k,len] );
  q = Q((l+1):L);
  q = reshape( q, [abcSize^k,len] );
  q = q - repmat( mean(q,1), abcSize^k, 1 );
  q( :, (len-k+2):len ) = 0;
  x{k} = q;
  l = L;
  X(k,:) = max( abs(x{k}), [], 1 );
  W(k,:) = max( abs(w{k}), [], 1 );
  %X(k,:) = var( x{k} );
end;
%save( 'S.mat', 'x', 'X' );


% === compute differential POIMs
%%% y = {};
%%% Y = zeros( max_order, len );
%%% for( k = 1:max_order )
%%%   if( k == 1 )
%%%     y{k} = x{k};
%%%   else
%%%     s = x{ k-1 };
%%%     i = (1:(abcSize^k)) - 1;
%%%     iR = floor( i / abcSize );
%%%     iL = mod( i, abcSize^(k-1) );
%%%     t = s( iR+1, 1:(len-1) );
%%%     t = max( t, s( iL+1, 2:len ) );
%%%     t( :, (len-k+2):len ) = zeros( abcSize^k, k-1 );
%%%     y{k} = x{k} - t;
%%%   end;
%%%   Y(k,:) = max( abs(y{k}), [], 1 );
%%% end;
Y = zeros( size(X) );
Y(1,:) = X(1,:);
for( k = 2:max_order )
  Y(k,1:(len-k+1)) = X(k,1:(len-k+1)) - max( X(k-1,1:(len-k+1)), X(k-1,2:(len-k+2)) );
end;


% === output
figure;
for( k = 1:4 )
  subplot( 2, 2, k );
  imagesc( x{k} );
  title( sprintf( 'POIM %d (shogun)', k ) );
end;
figure;
imagesc( X );
title( 'master POIM (shogun)' );
figure;
imagesc( Y );
title( 'master diff-POIM (shogun)' );
figure;
imagesc( W );
title( 'max w' );
if( 0 )
  figure;
  imagesc( w{1} );
  title( 'w, k=1' );
  figure;
  imagesc( w{2} );
  title( 'w, k=2' );
  figure;
  imagesc( w{3} );
  title( 'w, k=3' );
end;


% === predict all possible sequences
N = abcSize^len;
T = repmat( ' ', len, N );
P = zeros( N, len );
t = (1:N) - 1;
for( i = len:-1:1 )
  T(i,:) = acgt( mod(t,abcSize)+1 );
  P(:,i) = distrib( mod(t,abcSize)+1, i );
  t = floor( t / abcSize );
end;
P = prod( P, 2 );
sg( 'set_features', 'TEST', T, 'DNA' );
sg( 'set_labels', 'TEST', ones(1,N) );
out = sg( 'classify' );


% === compute true POIMs
poims = {};
meanOut = out * P;
%for( k = 1:max_order )
for( k = 1:4 )
  m = abcSize^k;
  poim = zeros( m, len );
  t = (1:N) - 1;
  for( i = (len-k+1):-1:1 )
    y = mod( t, m ) + 1;
    for( z = 1:m )
      poim(z,i) = out(y==z) * P(y==z) / sum(P(y==z));
    end;
    t = floor( t / abcSize );
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


