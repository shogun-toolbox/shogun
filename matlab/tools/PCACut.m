function [T,zeigen]=pcacut(Z, thresh) ;

%calc mean
mw=mean(Z,2) ;

%calc covariance matrix
covz=cov(Z', 1);

[v, d] = eig(covz);			% get the eigensystem,
                                        % negative eigenvalues have
                                        % to go ...
d = diag(d);				% cut out diagonal
[d,idx] = sort(d);
v=v(:,idx);
dgood = find(d > thresh) ;

fprintf('reducing to %i dimensions (EV:%e-%e)\n', length(dgood), min(d(dgood)), max(d(dgood))) ;
dinv = spdiags(d(dgood).^(-1/2), 0, length(dgood), length(dgood));

T=dinv*v(:,dgood)' ;

%fprintf('done') ;

zeigen = T * Z;

%zeigen = dinv * zeigen(dgood,:);

