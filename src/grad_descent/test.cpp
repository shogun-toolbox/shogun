#include "hmm/HMM.h"

CHMM* hmmcom ;
double linmin(REAL p[], REAL xi[], int n) ;

double get_objective(CHMM* pos)
{
  //int no_of_examples=pos->get_observations()->get_DIMENSION() ;

  double lik=exp( pos->model_probability(-1) * pos->get_observations()->get_DIMENSION()) ;
  //for (int num=0; num<no_of_examples; num++)
  //    lik+=pos->model_probability(num);
  //lik/=no_of_examples ;

  return lik ;

} ;

void get_gradient_vector(CHMM* pos, REAL* gradient)
{
  int no_of_examples=pos->get_observations()->get_DIMENSION() ;

  //pos->check_model_derivatives() ;
  //pos->check_model_derivatives_combined() ;

  for (int num=0; num<no_of_examples; num++)
    {
      int i,j,p=0;
      double prob_num=pos->model_probability(num);

      for (i=0; i<pos->get_N(); i++)
	gradient[p++] += exp(pos->model_derivative_p(i, num)-prob_num);
      
      for (i=0; i<pos->get_N(); i++)
	gradient[p++] += exp(pos->model_derivative_q(i, num)-prob_num);
      
      for (i=0; i<pos->get_N(); i++)
	for (j=0; j<pos->get_N(); j++) {
	  gradient[p++] += exp(pos->model_derivative_a(i, j, num)-prob_num);
	}
      for (i=0; i<pos->get_N(); i++)
	for (j=0; j<pos->get_M(); j++) {
	  gradient[p++] += exp(pos->model_derivative_b(i, j, num)-prob_num);
	} 
    } 
} 

void get_param_vector(CHMM* pos, REAL* params)
{
  long i,j,p=0;
#define DO_EXP(x) (x)

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
  long int len=pos->get_N()*(1+pos->get_N()+1+pos->get_M()) ;
  long int i ;
  params=new double[len] ;
  gradient=new double[len] ;
  hmmcom=pos ;

  for (i=0; i<len; i++) 
    gradient[i]=0 ;

  pos->invalidate_model() ;
  double lik = get_objective(pos) ;
  get_param_vector(pos, params) ;
  get_gradient_vector(pos, gradient) ;

  //REAL step=linmin(params, gradient, len) ;
  //printf("step=%e\n",step) ;

  for (i=0; i<len; i++)
    {
      //CIO::message("p[%i]=%e   %1.2e  g[%i]=%e\n", i, exp(params[i]), step, i, gradient[i]) ;
      params[i]+=step_size*gradient[i] ;
    }
  set_param_vector(pos, params) ;
  if (beta>0)
    pos->normalize() ;

  pos->invalidate_model() ;

  double lik2 = get_objective(pos) ;

  CIO::message("lik_before=%e  lik_after=%e\n", lik, lik2) ;
  delete[] params ;
  delete[] gradient ;
}
