function T=cleaner(covz, thresh) ;

fprintf('computing PCA in %i dimensions\n', size(covz,1)) ;

%covz = Z * Z';
%covz = covz / size(Z,2);			% rescale with sample size

d = eig(covz)

[v, d] = eig(covz);			% get the eigensystem,
                                        % negative eigenvalues have
                                        % to go ...
d = diag(d);				% cut out diagonal
dgood = d > thresh ;

fprintf('reducing to %i dimensions (EV:%e-%e)\n', sum(dgood), min(d(dgood)), max(d(dgood))) ;
dinv = spdiags(d(dgood).^(-1/2), 0, sum(dgood), sum(dgood));

T=dinv*v(:,dgood)' ;

%fprintf('done') ;

%zeigen = v' * Z;
%zeigen = dinv * zeigen(dgood,:);

