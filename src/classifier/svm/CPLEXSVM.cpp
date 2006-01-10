#include "classifier/svm/CPLEXSVM.h"
#include "lib/common.h"
#ifdef USE_CPLEX
#include "lib/io.h"
#include "lib/Mathmatics.h"
#include "lib/Cplex.h"
#include "features/Labels.h"

CCPLEXSVM::CCPLEXSVM()
{
}

CCPLEXSVM::~CCPLEXSVM()
{
}

bool CCPLEXSVM::train()
{
	bool result = false;
	CLabels* lab = CKernelMachine::get_labels();
	CCplex cplex;

	if (cplex.init_cplex(CCplex::QP))
	{
		INT n,m;
		INT num_label=0;
		REAL* y = lab->get_labels(num_label);
		REAL* H = kernel->get_kernel_matrix_real(m, n, NULL);
		assert(n>0 && n==m && n==num_label);
		REAL* alphas=new REAL[n];
		REAL* lb=new REAL[n];
		REAL* ub=new REAL[n];
		assert(lb && ub);

		//hessian y'y.*K
		for (int i=0; i<n; i++)
		{
			lb[i]=0;
			ub[i]=get_C1();

			for (int j=0; j<n; j++)
				H[i*n+j]*=y[j]*y[i];
		}

		//feed qp to cplex


		int j=0;
		for (int i=0; i<n; i++)
		{
			if (alphas[i]>0)
			{
				//set_alpha(j, alphas[i]*lab->get_label(i)/etas[1]);
				set_alpha(j, alphas[i]*lab->get_label(i));
				set_support_vector(j, i);
				j++;
			}
		}
		compute_objective();
		CIO::message(M_INFO, "obj = %.16f, rho = %.16f\n",get_objective(),get_bias());
		CIO::message(M_INFO, "Number of SV: %ld\n", get_num_support_vectors());

		delete[] alphas;
		delete[] lb;
		delete[] ub;
		delete[] H;
		delete[] y;

		result = true;
	}

	if (!result)
		CIO::message(M_ERROR, "cplex svm failed");

	return result;
}
#endif
