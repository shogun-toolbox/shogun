/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*   Copyright: Thorsten Joachims                                       */
/*   Date: 16.12.97                                                     */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. */

#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "svm/svm_common.h"
#include "hmm/HMM.h"
#include "lib/Observation.h"

extern CHMM* pos;
extern CHMM* neg;
extern double* theta;

double normalizer=1;

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
	int i,j;

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

		for (i=0; i<pos->get_N(); i++)
		{
		    if (*obs_x==*obs_y)
			result+=(exp(-pos->get_b(i, *obs_x))-exp(-neg->get_b(i, *obs_x)))*
			    (exp(-pos->get_b(i, *obs_y))-exp(-neg->get_b(i, *obs_y)));
		    obs_x++;
		    obs_y++;
		}

		/*int p=0;
		for (i=0; i<pos->get_N(); i++)
		{
			for (j=0; j<pos->get_M(); j++)
				theta[p++]=exp(pos->linear_model_derivative(i, j, x)-posx);
		}
		for (i=0; i<neg->get_N(); i++)
		{
			for (j=0; j<neg->get_M(); j++)
				theta[p++]=exp(neg->linear_model_derivative(i, j, x)-negx);
		}
		p=0;
		for (i=0; i<pos->get_N(); i++)
		{
			for (j=0; j<pos->get_M(); j++)
				result+=theta[p++]*exp(pos->linear_model_derivative(i, j, y)-posy);
		}
		for (i=0; i<neg->get_N(); i++)
		{
			for (j=0; j<neg->get_M(); j++)
				result+=theta[p++]*exp(neg->linear_model_derivative(i, j, y)-negy);
		}*/
	}
	return result/normalizer;
}


double top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b) /* plug in your favorite kernel */
{
	double result=0;

	int x=a->docnum;
	int y=b->docnum;
	int i,j;

	//calculate TOP Kernel
	if (x>=0 && y>=0)
	{
		double posx=pos->model_probability(x);
		double posy=pos->model_probability(y);
		double negx=neg->model_probability(x);
		double negy=neg->model_probability(y);

		result=(posx-negx)*(posy-negy);
		
		int p=0;
		for (i=0; i<pos->get_N(); i++)
		{
			theta[p++]=exp(pos->model_derivative_p(i, x)-posx);
			theta[p++]=exp(pos->model_derivative_q(i, x)-posx);

			for (j=0; j<pos->get_N(); j++)
				theta[p++]=exp(pos->model_derivative_a(i, j, x)-posx);

			for (j=0; j<pos->get_M(); j++)
				theta[p++]=exp(pos->model_derivative_b(i, j, x)-posx);
		}
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
		for (i=0; i<pos->get_N(); i++)
		{
			result+=theta[p++]*exp(pos->model_derivative_p(i, y)-posy);
			result+=theta[p++]*exp(pos->model_derivative_q(i, y)-posy);

			for (j=0; j<pos->get_N(); j++)
				result+=theta[p++]*exp(pos->model_derivative_a(i, j, y)-posy);

			for (j=0; j<pos->get_M(); j++)
				result+=theta[p++]*exp(pos->model_derivative_b(i, j, y)-posy);
		}
		for (i=0; i<neg->get_N(); i++)
		{
			result+=theta[p++]*exp(neg->model_derivative_p(i, y)-negy);
			result+=theta[p++]*exp(neg->model_derivative_q(i, y)-negy);

			for (j=0; j<neg->get_N(); j++)
				result+=theta[p++]*exp(neg->model_derivative_a(i, j, y)-negy);

			for (j=0; j<neg->get_M(); j++)
				result+=theta[p++]*exp(neg->model_derivative_b(i, j, y)-negy);
		}
	}
	return result/normalizer;
}
#endif
