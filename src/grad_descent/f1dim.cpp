#define NRANSI
#include "nrutil.h"
#include "distributions/hmm/HMM.h"

extern int ncom;
extern REAL *pcom,*xicom, *xt ;
extern CHMM* hmmcom ;

double get_objective(CHMM* pos) ;
void set_param_vector(CHMM* pos, REAL* params) ;

REAL f1dim(REAL x)
{
	int j;
	
	for (j=0;j<ncom;j++) 
	  xt[j] = pcom[j] + x*xicom[j];
	
	set_param_vector(hmmcom, xt) ;
	hmmcom->normalize() ;
	hmmcom->invalidate_model() ;

	REAL obj=-get_objective(hmmcom) ;

	printf("delta=%e  objective=%e\n",x,obj) ;
	return obj ;
}
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software .). */
