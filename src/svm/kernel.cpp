#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "svm/kernel.h"
#include "hmm/HMM.h"

extern CHMM* pos;
extern CHMM* neg;
extern double* theta;
extern double* featurespace;
extern int num_features;

long   kernel_cache_statistic;
double normalizer=1;

/* calculate the kernel function */
CFLOAT kernel(KERNEL_PARM *kernel_parm,DOC* a,DOC* b)
{
  
    kernel_cache_statistic++;

    if (a->docnum < 0 || b ->docnum <0)
    {
#ifdef DEBUG
	printf("ERROR: (%d,%d)\n", a->docnum, b->docnum);
#endif
	return 0;
    }

  switch(kernel_parm->kernel_type) {
////    case 0: /* linear */ 
////            return((CFLOAT)sprod_ss(a->words,b->words)); 
////    case 1: /* polynomial */
////            return((CFLOAT)pow(kernel_parm->coef_lin*sprod_ss(a->words,b->words)+kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
////    case 2: /* radial basis function */
////            return((CFLOAT)exp(-kernel_parm->rbf_gamma*(a->twonorm_sq-2*sprod_ss(a->words,b->words)+b->twonorm_sq)));
////    case 3: /* sigmoid neural net */
////            return((CFLOAT)tanh(kernel_parm->coef_lin*sprod_ss(a->words,b->words)+kernel_parm->coef_const)); 
		case 4: /* TOP Kernel */
	        return((CFLOAT)top_kernel(kernel_parm,a,b)); 
		case 5: 
			return ((CFLOAT)linear_top_kernel(kernel_parm,a,b)); 
		case 6:
			return ((CFLOAT)cached_top_kernel(kernel_parm,a,b)); 

	    default: printf("Error: Unknown kernel function\n"); exit(1);
	}
}

void tester(KERNEL_PARM *kernel_parm)
{
	DOC a;
	DOC b;

	double kern[139][139];
	int i;
	for (i=0; i<139; i++)
	{
		for (int j=i; j<139; j++)
		{
			a.docnum=i;
			b.docnum=j;
			double v1=kernel(kernel_parm,&a,&b);

			a.docnum=i;
			b.docnum=j;
			double v2=kernel(kernel_parm,&a,&b);

			if (v2!=v1)
				printf("ERRROOOOOORR (%d,%d) -> %e|%e\n",i,j,v1,v2);

			kern[i][j]=v1;
			kern[j][i]=v2;
		}
	}

	for (i=0; i<139; i++)
	{
		for (int j=i; j<139; j++)
		{
			a.docnum=i;
			b.docnum=j;
			double v1=kernel(kernel_parm,&a,&b);

			a.docnum=i;
			b.docnum=j;
			double v2=kernel(kernel_parm,&a,&b);

			if (v2!=v1 || kern[i][j]-v1 || kern[j][i]!=v2)
				printf("ERRROOOOOORR\n\t\t (%d,%d) -> %e|%e %e|%e\n",i,j,kern[i][j],v1,kern[j][i],v2);
		}
	}
}

double find_normalizer(KERNEL_PARM *kernel_parm, int num)
{
	DOC a;
	double sum=0;
	normalizer=1.0;

	for (int i=0; i<num; i++)
	{
		a.docnum=i;
		sum+=kernel(kernel_parm, &a, &a);
	}
	normalizer=sum/num;

	return normalizer;
}

double linear_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b) /* plug in your favorite kernel */
{
	double result=0;

	int x=a->docnum;
	int y=b->docnum;

	//calculate TOP Kernel
	if (x>=0 && y>=0)
	{
		double posx=pos->linear_model_probability(x);
		double posy=pos->linear_model_probability(y);
		double negx=neg->linear_model_probability(x);
		double negy=neg->linear_model_probability(y);

		result=(posx-negx)*(posy-negy);

		T_OBSERVATIONS* obs_x=(pos->get_observations())->get_obs_vector(x);
		T_OBSERVATIONS* obs_y=(pos->get_observations())->get_obs_vector(y);

		for (int i=0; i<pos->get_N(); i++)
		{
		    if (*obs_x==*obs_y)
			result+=(exp(-pos->get_b(i, *obs_x))-exp(-neg->get_b(i, *obs_x)))*
			    (exp(-pos->get_b(i, *obs_y))-exp(-neg->get_b(i, *obs_y)));
		    obs_x++;
		    obs_y++;
		}

	}
	return result/normalizer;
}


double top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b) /* plug in your favorite kernel */
{
    double result=0;

    int x=a->docnum;
    int y=b->docnum;

    int i,j,p=0;
    double posx=pos->model_probability(x);
    double negx=neg->model_probability(x);
    
    theta[p++]=(posx-negx);

    //first do all derivatives of parameters of positive model for sequence x
    for (i=0; i<pos->get_N(); i++)
    {
	theta[p++]=exp(pos->model_derivative_p(i, x)-posx);
	theta[p++]=exp(pos->model_derivative_q(i, x)-posx);

	for (j=0; j<pos->get_N(); j++)
	    theta[p++]=exp(pos->model_derivative_a(i, j, x)-posx);

	for (j=0; j<pos->get_M(); j++)
	    theta[p++]=exp(pos->model_derivative_b(i, j, x)-posx);

    }
    
    //then do all derivatives of parameters of negative model for sequence y
    for (i=0; i<neg->get_N(); i++)
    {
	theta[p++]=exp(neg->model_derivative_p(i, x)-negx);
	theta[p++]=exp(neg->model_derivative_q(i, x)-negx);

	for (j=0; j<neg->get_N(); j++)
	    theta[p++]=exp(neg->model_derivative_a(i, j, x)-negx);

	for (j=0; j<neg->get_M(); j++)
	    theta[p++]=exp(neg->model_derivative_b(i, j, x)-negx);

    }

    p=0;  
    double posy=pos->model_probability(y);
    double negy=neg->model_probability(y);

    result=theta[p++]*(posy-negy);

    //second do all derivatives of parameters of positive model for sequence y
    for (i=0; i<pos->get_N(); i++)
    {
	result+=theta[p++]*exp(pos->model_derivative_p(i, y)-posy);
	result+=theta[p++]*exp(pos->model_derivative_q(i, y)-posy);

	for (j=0; j<pos->get_N(); j++)
	    result+=theta[p++]*exp(pos->model_derivative_a(i, j, y)-posy);

	for (j=0; j<pos->get_M(); j++)
	    result+=theta[p++]*exp(pos->model_derivative_b(i, j, y)-posy);
    }

    //... and last derivatives of parameters of negative model for sequence y
    for (i=0; i<neg->get_N(); i++)
    {
	result+=theta[p++]*exp(neg->model_derivative_p(i, y)-negy);
	result+=theta[p++]*exp(neg->model_derivative_q(i, y)-negy);

	for (j=0; j<neg->get_N(); j++)
	    result+=theta[p++]*exp(neg->model_derivative_a(i, j, y)-negy);

	for (j=0; j<neg->get_M(); j++)
	    result+=theta[p++]*exp(neg->model_derivative_b(i, j, y)-negy);
    }

    return result/normalizer;
}

double cached_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b) /* plug in your favorite kernel */
{
    double* features_a=&featurespace[num_features*a->docnum];
    double* features_b=&featurespace[num_features*b->docnum];
    double result=0;
    int i=num_features;

    while (i--)
	result+= *features_a++ * *features_b++;

    result/=normalizer;
#ifdef KERNEL_DEBUG
    double top_res=top_kernel(kernel_parm,a,b);
    if (fabs(top_res-result)>1e-6)
	printf("cached kernel bug:%e == %e\n", top_kernel(kernel_parm,a,b), result);
#endif
    return result;
}
#endif
