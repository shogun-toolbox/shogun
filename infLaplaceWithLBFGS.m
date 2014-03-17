function [post nlZ dnlZ] = infLaplaceWithLBFGS(hyp, mean, cov, lik, x, y, opt)

% Laplace approximation to the posterior Gaussian process.
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likFunction.m), and is designed to be used with
% gp.m. See also infFunctions.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-05-02.
%
% See also INFMETHODS.M.

persistent last_alpha                                   % copy of the last alpha
if any(isnan(last_alpha)), last_alpha = zeros(size(last_alpha)); end   % prevent

if nargin<=6, opt = []; end                        % make opt variable available
if isfield(opt,'postL'), postL = opt.postL;        % recompute matrix L for post
else postL = true; end                                           % default value

inf = 'infLaplace';
n = size(x,1);
if isnumeric(cov),  K = cov;                    % use provided covariance matrix
else K = feval(cov{:},  hyp.cov,  x); end       % evaluate the covariance matrix
if isnumeric(mean), m = mean;                         % use provided mean vector
else m = feval(mean{:}, hyp.mean, x); end             % evaluate the mean vector
likfun = @(f) feval(lik{:},hyp.lik,y,f,[],inf);        % log likelihood function

if any(size(last_alpha)~=[n,1])     % find a good starting point for alpha and f
  alpha = zeros(n,1);                      % start at mean if sizes do not match
else
  alpha = last_alpha;                                             % try last one
  if Psi(alpha,m,K,likfun) > -sum(likfun(m))     % default f==m better => use it
    alpha = zeros(n,1);
  end
end

% switch between optimisation methods
%alpha = irls(alpha, m,K,likfun, opt);                         % run optimisation
disp('called LBFGS')
alpha = lbfgs(alpha, m,K,likfun, opt);

f = K*alpha+m;                                  % compute latent function values
last_alpha = alpha;                                     % remember for next call
[lp,dlp,d2lp,d3lp] = likfun(f); W = -d2lp; isWneg = any(W<0);
post.alpha = alpha;                            % return the posterior parameters
post.sW = sqrt(abs(W)).*sign(W);             % preserve sign in case of negative

% diagnose optimality
err = @(x,y) norm(x-y)/max([norm(x),norm(y),1]);   % we need to have alpha = dlp
% dev = err(alpha,dlp);  if dev>1e-4, warning('Not at optimum %1.2e.',dev), end

if postL || nargout>1
  if isWneg                  % switch between Cholesky and LU decomposition mode
    % For post.L = -inv(K+diag(1./W)), we us the non-default parametrisation.
    [ldA, iA, post.L] = logdetA(K,W);   % A=eye(n)+K*W is as safe as symmetric B
    nlZ = alpha'*(f-m)/2 - sum(lp) + ldA/2;
  else
    sW = post.sW; post.L = chol(eye(n)+sW*sW'.*K);                   % recompute
    nlZ = alpha'*(f-m)/2 + sum(log(diag(post.L))-lp);   % ..(f-m)/2 -lp +ln|B|/2
  end
end

if nargout>2                                           % do we want derivatives?
  dnlZ = hyp;                                   % allocate space for derivatives
  if isWneg                  % switch between Cholesky and LU decomposition mode
    Z = -post.L;                                                 % inv(K+inv(W))
    g = sum(iA.*K,2)/2; % deriv. of ln|B| wrt W; g = diag(inv(inv(K)+diag(W)))/2
  else
    Z = repmat(sW,1,n).*solve_chol(post.L,diag(sW)); %sW*inv(B)*sW=inv(K+inv(W))
    C = post.L'\(repmat(sW,1,n).*K);                     % deriv. of ln|B| wrt W
    g = (diag(K)-sum(C.^2,1)')/2;                    % g = diag(inv(inv(K)+W))/2
  end
  dfhat = g.*d3lp;  % deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
  for i=1:length(hyp.cov)                                    % covariance hypers
    dK = feval(cov{:}, hyp.cov, x, [], i);
    dnlZ.cov(i) = sum(sum(Z.*dK))/2 - alpha'*dK*alpha/2;         % explicit part
    b = dK*dlp;                            % b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
    dnlZ.cov(i) = dnlZ.cov(i) - dfhat'*( b-K*(Z*b) );            % implicit part
  end
  for i=1:length(hyp.lik)                                    % likelihood hypers
    [lp_dhyp,dlp_dhyp,d2lp_dhyp] = feval(lik{:},hyp.lik,y,f,[],inf,i);
    dnlZ.lik(i) = -g'*d2lp_dhyp - sum(lp_dhyp);                  % explicit part
    b = K*dlp_dhyp;                        % b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
    dnlZ.lik(i) = dnlZ.lik(i) - dfhat'*( b-K*(Z*b) );            % implicit part
  end
  for i=1:length(hyp.mean)                                         % mean hypers
    dm = feval(mean{:}, hyp.mean, x, i);
    dnlZ.mean(i) = -alpha'*dm;                                   % explicit part
    dnlZ.mean(i) = dnlZ.mean(i) - dfhat'*(dm-K*(Z*dm));          % implicit part
  end
end

% Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where 
% f = K*alpha+m, and likfun(f) = feval(lik{:},hyp.lik,y,  f,  [],inf).
function [psi,dpsi,f,alpha,dlp,W] = Psi(alpha,m,K,likfun)
  f = K*alpha+m;
  [lp,dlp,d2lp] = likfun(f); W = -d2lp;
  psi = alpha'*(f-m)/2 - sum(lp);
  if nargout>1, dpsi = K*(alpha-dlp); end

function [psi, g] = lbfgs_helper(alpha, m,K,likfun, opt)
	psi = Psi(alpha,m,K,likfun);
	f = K*alpha+m; [lp,dlp] = likfun(f); 
	g = K*(alpha-dlp);

function alpha = lbfgs(alpha, m,K,likfun, opt)
	optMinFunc = struct('Display', 'FULL',...
    'Method', 'lbfgs',...
    'DerivativeCheck', 'off',...
    'LS_type', 1,...
    'MaxIter', 1000,...
	'LS_interp', 1,...
    'MaxFunEvals', 1000000,...
    'Corr' , 100,...
    'optTol', 1e-15,...
    'progTol', 1e-15);
	[alpha, psi_new] = minFunc(@lbfgs_helper, alpha, optMinFunc, m,K,likfun);


% Run IRLS Newton algorithm to optimise Psi(alpha).
function alpha = irls(alpha, m,K,likfun, opt)
  if isfield(opt,'irls_maxit'), maxit = opt.irls_maxit; % max no of Newton steps
  else maxit = 20; end                                           % default value
  if isfield(opt,'irls_Wmin'),  Wmin = opt.irls_Wmin; % min likelihood curvature
  else Wmin = 0.0; end                                           % default value
  if isfield(opt,'irls_tol'),   tol = opt.irls_tol;     % stop Newton iterations
  else tol = 1e-6; end                                           % default value

  smin_line = 0; smax_line = 2;           % min/max line search steps size range
  nmax_line = 10;                          % maximum number of line search steps
  thr_line = 1e-4;                                       % line search threshold
  Psi_line = @(s,alpha,dalpha) Psi(alpha+s*dalpha, m,K,likfun);    % line search
  pars_line = {smin_line,smax_line,nmax_line,thr_line};  % line seach parameters
  search_line = @(alpha,dalpha) brentmin(pars_line{:},Psi_line,5,alpha,dalpha);

  f = K*alpha+m; [lp,dlp,d2lp] = likfun(f); W = -d2lp; n = size(K,1);
  Psi_new = Psi(alpha,m,K,likfun);
  Psi_old = Inf;  % make sure while loop starts by the largest old objective val
  it = 0;                          % this happens for the Student's t likelihood
  while Psi_old - Psi_new > tol && it<maxit                       % begin Newton
    Psi_old = Psi_new; it = it+1;
    % limit stepsize
    W = max(W,Wmin); % reduce step size by increasing curvature of problematic W
    sW = sqrt(W); L = chol(eye(n)+sW*sW'.*K);            % L'*L=B=eye(n)+sW*K*sW
    b = W.*(f-m) + dlp;
    dalpha = b - sW.*solve_chol(L,sW.*(K*b)) - alpha; % Newton dir + line search
    [s_line,Psi_new,n_line,dPsi_new,f,alpha,dlp,W] = search_line(alpha,dalpha);
  end                                                  % end Newton's iterations

% Compute the log determinant ldA and the inverse iA of a square nxn matrix
% A = eye(n) + K*diag(w) from its LU decomposition; for negative definite A, we 
% return ldA = Inf. We also return mwiA = -diag(w)/A.
function [ldA,iA,mwiA] = logdetA(K,w)
  [m,n] = size(K); if m~=n, error('K has to be nxn'), end
  A = eye(n)+K.*repmat(w',n,1);
  [L,U,P] = lu(A); u = diag(U);           % compute LU decomposition, A = P'*L*U
  signU = prod(sign(u));                                             % sign of U
  detP = 1;                 % compute sign (and det) of the permutation matrix P
  p = P*(1:n)';
  for i=1:n                                                       % swap entries
    if i~=p(i), detP = -detP; j = find(p==i); p([i,j]) = p([j,i]); end
  end
  if signU~=detP  % log becomes complex for negative values, encoded by infinity
    ldA = Inf;
  else            % det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
    ldA = sum(log(abs(u)));
  end 
  if nargout>1, iA = U\(L\P); end               % return the inverse if required
  if nargout>2, mwiA = -repmat(w,1,n).*iA; end
