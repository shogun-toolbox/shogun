#include "hmm/HMM.h"

double get_objective(CHMM* pos)
{
  //int no_of_examples=pos->get_observations()->get_DIMENSION() ;

  double lik=exp( pos->model_probability(-1) * pos->get_observations()->get_DIMENSION()) ;
  //for (int num=0; num<no_of_examples; num++)
  //    lik+=pos->model_probability(num);
  //lik/=no_of_examples ;

  return lik ;

} ;

void get_gradient_vector(CHMM* pos, REAL* gradient, long& len)
{
  int no_of_examples=pos->get_observations()->get_DIMENSION() ;

  //  pos->check_model_derivatives() ;
  // pos->check_model_derivatives_combined() ;

  len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) ;  

  double prob_all=pos->model_probability(-1)*pos->get_observations()->get_DIMENSION();
  
  for (int num=0; num<no_of_examples; num++)
    {
      int i,j,p=0;
      double prob_num=pos->model_probability(num);

      //for (i=0; i<pos->get_N(); i++)
      //gradient[p++]=exp(pos->model_derivative_p(i, num)+prob_all-prob_num);
      
      //for (i=0; i<pos->get_N(); i++)
      //gradient[p++]=exp(pos->model_derivative_q(i, num)+prob_all-prob_num);
      
      for (i=0; i<pos->get_N(); i++)
	for (j=0; j<pos->get_N(); j++) {
	  gradient[p++]=exp(pos->model_derivative_a(i, j, num)+prob_all-prob_num);
	}
      
      //for (i=0; i<pos->get_N(); i++)
      //for (j=0; j<pos->get_M(); j++) {
      //  gradient[p++]=exp(pos->model_derivative_b(i, j, num)+prob_all-prob_num);
      //} 
    } 
} 

void project_gradient_vector(CHMM* pos, REAL* gradient)
{ 
  int p=0, old_p=0, i, j ;
  double sum=0 ;
  
  for (i=0; i<pos->get_N(); i++)
    sum+=gradient[p++] ;
  p=old_p ; 
  sum/=pos->get_N() ;
  for (i=0; i<pos->get_N(); i++)
    gradient[p++]-=sum ;
  
  old_p=p ; sum=0 ;
  for (i=0; i<pos->get_N(); i++)
    sum+=gradient[p++] ;
  p=old_p ; 
  sum/=pos->get_N() ;
  for (i=0; i<pos->get_N(); i++)
    gradient[p++]-=sum ;
  
  for (i=0; i<pos->get_N(); i++)
    {
      old_p=p ; sum=0 ;
      for (j=0; j<pos->get_N(); j++) 
	sum+=gradient[p++] ;
      p=old_p ; 
      sum/=pos->get_N() ;
      for (j=0; j<pos->get_N(); j++) 
	gradient[p++]-=sum ;
    }
  
  for (i=0; i<pos->get_N(); i++)
    {
      old_p=p ; sum=0 ;
      for (j=0; j<pos->get_M(); j++) 
	sum+=gradient[p++] ;
      p=old_p ; sum/=pos->get_M() ;
      for (j=0; j<pos->get_M(); j++) 
	gradient[p++]-=sum ;	
    } 
} 

void normalize_param_vector(CHMM* pos, REAL* params)
{ 
  int p=0, old_p=0, i, j ;
  double sum ;

  sum=0 ; old_p=0 ;
  for (i=0; i<pos->get_N(); i++)
    sum+=exp(params[p++]) ;
  p=old_p ; 
  for (i=0; i<pos->get_N(); i++)
    params[p++]-=log(sum) ;
  
  old_p=p ; sum=0 ;
  for (i=0; i<pos->get_N(); i++)
    sum+=exp(params[p++]) ;
  p=old_p ; 
  for (i=0; i<pos->get_N(); i++)
    params[p++]-=log(sum) ;
  
  for (i=0; i<pos->get_N(); i++)
    {
      old_p=p ; sum=0 ;
      for (j=0; j<pos->get_N(); j++) 
	sum+=exp(params[p++]) ;
      p=old_p ; 
      for (j=0; j<pos->get_N(); j++) 
	params[p++]-=log(sum) ;
    }
  
  for (i=0; i<pos->get_N(); i++)
    {
      old_p=p ; sum=0 ;
      for (j=0; j<pos->get_M(); j++) 
	sum+=exp(params[p++]) ;
      p=old_p ; 
      for (j=0; j<pos->get_M(); j++) 
	params[p++]-=log(sum) ;	
    } 
} 

void gradient_add_barrier(REAL* gradient, REAL* params, long len, double beta)
{
  for (int i=0; i<len; i++)
    gradient[i]+=exp(-params[i]/beta) ;
} ;

void get_param_vector(CHMM* pos, REAL* params, long& len)
{
  long i,j,p=0;
#define DO_EXP(x) (x)

  len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) ;
  
  for (i=0; i<pos->get_N(); i++)
    params[p++]=DO_EXP(pos->get_p(i)) ;
  
  for (i=0; i<pos->get_N(); i++)
    params[p++]=DO_EXP(pos->get_q(i)) ;
  
  for (i=0; i<pos->get_N(); i++)
    for (j=0; j<pos->get_N(); j++) {
      params[p++]=DO_EXP(pos->get_a(i, j)) ;
    }
  
  for (i=0; i<pos->get_N(); i++)
    for (j=0; j<pos->get_M(); j++) {
      params[p++]=DO_EXP(pos->get_b(i, j)) ;
    } 
  
} ;

void set_param_vector(CHMM* pos, REAL* params)
{
#define DO_LOG(x) (x)
  long i,j,p=0;
  
  for (i=0; i<pos->get_N(); i++)
    pos->set_p(i,DO_LOG(params[p++])) ;
  
  for (i=0; i<pos->get_N(); i++)
    pos->set_q(i,DO_LOG(params[p++])) ;
  
  for (i=0; i<pos->get_N(); i++)
    for (j=0; j<pos->get_N(); j++) {
      pos->set_a(i, j, DO_LOG(params[p++])) ;
    }
  
  for (i=0; i<pos->get_N(); i++)
    for (j=0; j<pos->get_M(); j++) {
      pos->set_b(i, j, DO_LOG(params[p++])) ;
    } 
  pos->invalidate_model() ;  
} ;

void fixed_descent(CHMM* pos, REAL step_size, REAL beta) 
{
  REAL * params, * gradient ;
  long int len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) ;
  long int i ;
  params=new double[len] ;
  gradient=new double[len] ;

  double lik = get_objective(pos) ;
  get_param_vector(pos, params, len) ;
  get_gradient_vector(pos, gradient, len) ;
  //gradient_add_barrier(gradient, params, len, beta) ;
  //project_gradient_vector(pos, gradient) ;

  for (i=0; i<len; i++)
    {
      CIO::message("p[%i]=%1.2e   %1.2e  g[%i]=%1.2e\n", i, params[i], step_size, i, gradient[i]) ;
      params[i]+=step_size*gradient[i] ;
    }
  set_param_vector(pos, params) ;
  pos->normalize() ;
  //normalize_param_vector(pos, params) ;

  double lik2 = get_objective(pos) ;

  CIO::message("lik_before=%1.3e  lik_after=%1.3e\n", lik, lik2) ;
  delete[] params ;
  delete[] gradient ;
}
