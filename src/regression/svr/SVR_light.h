#ifndef _SVRLight_H___
#define _SVRLight_H___

#include "classifier/svm/SVM_light.h"
class CSVRLight:public CSVMLight
{
 public:
  CSVRLight();
  virtual ~CSVRLight() {};
  
  virtual bool	train();
  void   svr_learn();

  virtual double compute_objective_function(double *a, double *lin, double *c, double eps, INT *label, INT totdoc);
  virtual void update_linear_component(LONG* docs, INT *label, 
							   long int *active2dnum, double *a, double* a_old,
							   long int *working2dnum, long int totdoc,
							   double *lin, DREAL *aicache, double* c);
  // MKL stuff
  virtual void update_linear_component_mkl(LONG* docs, INT *label, 
								   long int *active2dnum, double *a, double* a_old,
								   long int *working2dnum, long int totdoc,
								   double *lin, DREAL *aicache, double* c);
  virtual void update_linear_component_mkl_linadd(LONG* docs, INT *label, 
										  long int *active2dnum, double *a, double* a_old,
										  long int *working2dnum, long int totdoc,
										  double *lin, DREAL *aicache, double* c);

  virtual void   reactivate_inactive_examples(INT *label,double *a,SHRINK_STATE *shrink_state,
				      double *lin, double *c, long int totdoc,long int iteration,
				      long int *inconsistent,
				      long int *docs,MODEL *model,DREAL *aicache,
				      double* maxdiff) ;
protected:
static void* update_linear_component_linadd_helper(void *params);

inline INT regression_fix_index(INT i)
{
	if (i>=num_vectors)
		i=2*num_vectors-1-i;

	return i;
}

static inline INT regression_fix_index2(INT i, INT num_vectors)
{
	if (i>=num_vectors)
		i=2*num_vectors-1-i;

	return i;
}

inline virtual DREAL compute_kernel(INT i, INT j)
{
	i=regression_fix_index(i);
	j=regression_fix_index(j);

	if (use_precomputed_subkernels)
	{
		if (j>i)
			CMath::swap(i,j) ;
		DREAL sum=0 ;
		INT num_weights=-1 ;

		const DREAL * w = CKernelMachine::get_kernel()->get_subkernel_weights(num_weights) ;
		for (INT n=0; n<num_precomputed_subkernels; n++)
			if (w[n]!=0)
				sum += w[n]*precomputed_subkernels[n][i*(i+1)/2+j] ;
		return sum ;
	}
	else
		return CKernelMachine::get_kernel()->kernel(i, j) ;
}

  INT num_vectors; //number of train elements
};
#endif
