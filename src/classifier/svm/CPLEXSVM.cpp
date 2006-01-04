#include "classifier/svm/CPLEXSVM.h"
#include "lib/io.h"
#include "lib/common.h"
#include "lib/Mathmatics.h"

CCPLEXSVM::CCPLEXSVM()
{
}

CCPLEXSVM::~CCPLEXSVM()
{
}

bool CCPLEXSVM::train()
{
	CLabels* lab = CKernelMachine::get_labels();


	//const REAL nu=0.32;
	const REAL alpha_eps=1e-12;
	const REAL eps=get_epsilon();
	const long int maxiter = 1L<<30;
	//const bool nustop=false;
	//const int k=2;
	const int n=lab->get_num_labels();
	//const REAL d = 1.0/n/nu; //NUSVC
	const REAL d = get_C1(); //CSVC
	const REAL primaleps=eps;
	const REAL dualeps=eps*n; //heuristic
	long int niter=0;

	//kernel_cache = new CCache<KERNELCACHE_ELEM>(kernel->get_cache_size(), n, n);
	REAL* alphas=new REAL[n];
	REAL* dalphas=new REAL[n];
	//REAL* hessres=new REAL[2*n];
	REAL* hessres=new REAL[n];
	//REAL* F=new REAL[2*n];
	REAL* F=new REAL[n];

	//REAL hessest[2]={0,0};
	//REAL hstep[2];
	//REAL etas[2]={0,0};
	//REAL detas[2]={0,1}; //NUSVC
	REAL etas=0;
	REAL detas=0;   //CSVC
	REAL hessest=0;
	REAL hstep;

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
	delete[] dalphas;
	delete[] hessres;
	delete[] F;

	return true;
}
