function [T,zeigen]=cleaner(Z, thresh) ;


fprintf('computing PCA in %i dimensions\n', size(covz,1)) ;

m=mean(Z,2) ;
for i=1:size(Z,2)
     Z(:,i) = Z(:,i)-m ;
end ;

covz = (Z * Z';
covz = covz / size(Z,2);	       % rescale with sample size

%d = eig(covz)

[v, d] = eig(covz);			% get the eigensystem,
                                        % negative eigenvalues have
                                        % to go ...
d = diag(d);				% cut out diagonal
[d,idx] = sort(d) ;
v=v(:,idx) ;
dgood = find(d > thresh) ;

fprintf('reducing to %i dimensions (EV:%e-%e)\n', length(dgood), min(d(dgood)), max(d(dgood))) ;
dinv = spdiags(d(dgood).^(-1/2), 0, length(dgood), length(dgood));

T=dinv*v(:,dgood)' ;

%fprintf('done') ;

zeigen = T * Z;

%zeigen = dinv * zeigen(dgood,:);

