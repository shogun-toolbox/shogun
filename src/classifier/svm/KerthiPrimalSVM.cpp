/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/KerthiPrimalSVM.h"
#include "lib/io.h"
#include "lib/common.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CKerthiPrimalSVM::CKerthiPrimalSVM()
{
}

CKerthiPrimalSVM::~CKerthiPrimalSVM()
{
}

bool CKerthiPrimalSVM::train()
{

//	CLabels* lab = CSVM::get_labels();
//	const int n=lab->get_num_labels();
//	const DREAL* al_init=new DREAL[n];
//	const DREAL C=get_C1();

//function [al, b] = sparse_primal2(Y,C,al_init)
//  global X   % X is sparse and global to save memory
//             % size is n x d (n = nb of examples, d = nb of dimensions)
//  
//  prec = 1e-3;  % Relative precision both for the CG and the Newton optimization
//  itmax = 10; % Max number of iterations (also for CG and Newton)
//  
//  [n,d] = size(X);
//  
//  if nargin<3
//    al = zeros(n,1);
//  else
//    al = al_init;
//  end;
//  
//  it=0;
//  old_obj = +inf;
//  while 1
//    w = X'*al;
//    b = 0*sum(al);
//    out = Y.*(X*w+b) - 1;
//    sv = find(out<0);
//    obj = C*sum(out(sv).^2)/2 + (w'*w)/2; % L2 penalization of the errors
//    grad = w + C*X(sv,:)'*(out(sv).*Y(sv)); % Gradient
//    gradb = C*sum(out(sv).*Y(sv));
//    grad = X*grad + 0*gradb;
//    if obj>old_obj       % We should do back tracking or line search here
//      step_ = step_/2;   % But note that a full Newton step is usually very good.
//      fprintf('   Step reduced, Obj = %f\n',obj);
//    else
//      % step_ = hess \ grad;
//      % Solved with minres but other functions might be more stable
//      % and/or efficient.
//      [step_, foo, relres] = minres(@hess_vect_mult,grad,prec,itmax,[],[],[],sv,C);
//      it = it+1;
//      alold = al;
//      old_obj = obj;
//      fprintf('Iter = %d, Obj = %f, Step = %f, Nb of sv = %d,\t CG prec = %f\n',...
//              it,obj,grad'*step_/2,length(sv),relres);
//      if ((grad'*step_)<prec*obj) || (it>itmax) break; end;
//    end;
//    al = alold - step_;
//  end;
//  fprintf('\n');
//  
//  b = 0*sum(al);
//  
//
//function y = hess_vect_mult(x,sv,C)
//  % Compute the Hessian times a given vector x.
//  % hess = diag([ones(d-1,1); 0]) + C*(X(sv,:)'*X(sv,:));
//  global X
//  w = X'*x;
//  y = X*w;
//  z = C*(X(sv,:)*w+0*sum(x));
//  w = X(sv,:)'*z;
//  y = y + X*w+0*sum(z) + 2e0*x;
	return false;
}
