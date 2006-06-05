#include "classifier/svm/MPD.h"
#include "lib/io.h"
#include "lib/common.h"
#include "lib/Mathmatics.h"

CMPDSVM::CMPDSVM()
{
}

CMPDSVM::~CMPDSVM()
{
}

bool CMPDSVM::train()
{
	CLabels* lab = CKernelMachine::get_labels();

	//const DREAL nu=0.32;
	const DREAL alpha_eps=1e-12;
	const DREAL eps=get_epsilon();
	const long int maxiter = 1L<<30;
	//const bool nustop=false;
	//const int k=2;
	const int n=lab->get_num_labels();
	//const DREAL d = 1.0/n/nu; //NUSVC
	const DREAL d = get_C1(); //CSVC
	const DREAL primaleps=eps;
	const DREAL dualeps=eps*n; //heuristic
	long int niter=0;

	kernel_cache = new CCache<KERNELCACHE_ELEM>(kernel->get_cache_size(), n, n);
	DREAL* alphas=new DREAL[n];
	DREAL* dalphas=new DREAL[n];
	//DREAL* hessres=new DREAL[2*n];
	DREAL* hessres=new DREAL[n];
	//DREAL* F=new DREAL[2*n];
	DREAL* F=new DREAL[n];

	//DREAL hessest[2]={0,0};
	//DREAL hstep[2];
	//DREAL etas[2]={0,0};
	//DREAL detas[2]={0,1}; //NUSVC
	DREAL etas=0;
	DREAL detas=0;   //CSVC
	DREAL hessest=0;
	DREAL hstep;

	const DREAL stopfac = 1;

	bool primalcool;
	bool dualcool;

	//if (nustop)
		//etas[1] = 1;

	for (int i=0; i<n; i++)
	{
		alphas[i]=0;
		F[i]=lab->get_label(i);
		//F[i+n]=-1;
		hessres[i]=lab->get_label(i);
		//hessres[i+n]=-1;
		//dalphas[i]=F[i+n]*etas[1]; //NUSVC
		dalphas[i]=-1; //CSVC
	}

	// go ...
	while (niter++ < maxiter)
	{
		int maxpidx=-1;
		DREAL maxpviol = -1;
		//DREAL maxdviol = CMath::abs(detas[0]);
		DREAL maxdviol = CMath::abs(detas);
		bool free_alpha=false;

		//if (CMath::abs(detas[1])> maxdviol)
			//maxdviol=CMath::abs(detas[1]);

		// compute kkt violations with correct sign ...
		for (int i=0; i<n; i++)
		{
			DREAL v=CMath::abs(dalphas[i]);

			if (alphas[i] > 0 && alphas[i] < d)
				free_alpha=true;

			if ( (dalphas[i]==0) ||
					(alphas[i]==0 && dalphas[i] >0) ||
					(alphas[i]==d && dalphas[i] <0)
			   )
				v=0;

			if (v > maxpviol)
			{
				maxpviol=v;
				maxpidx=i;
			} // if we cannot improve on maxpviol, we can still improve by choosing a cached element
			else if (v == maxpviol) 
			{
				if (kernel_cache->is_cached(i))
					maxpidx=i;
			}
		}

		if (maxpidx<0 || maxdviol<0)
			CIO::message(M_ERROR, "no violation no convergence, should not happen!\n");

		// ... and evaluate stopping conditions
		//if (nustop)
			//stopfac = CMath::max(etas[1], 1e-10);    
		//else
			//stopfac = 1;

		if (niter%10000 == 0)
		{
			DREAL obj=0;

			for (int i=0; i<n; i++)
			{
				obj-=alphas[i];
				for (int j=0; j<n; j++)
					obj+=0.5*lab->get_label(i)*lab->get_label(j)*alphas[i]*alphas[j]*kernel->kernel(i,j);
			}

			CIO::message(M_DEBUG, "obj:%f pviol:%f dviol:%f maxpidx:%d iter:%d\n", obj, maxpviol, maxdviol, maxpidx, niter);
		}

		//for (int i=0; i<n; i++)
		//	CIO::message(M_DEBUG, "alphas:%f dalphas:%f\n", alphas[i], dalphas[i]);

		primalcool = (maxpviol < primaleps*stopfac);
		dualcool = (maxdviol < dualeps*stopfac) || (!free_alpha);

		// done?
		if (primalcool && dualcool)
		{
			if (!free_alpha)
				CIO::message(M_INFO, " no free alpha, stopping! #iter=%d\n", niter);
			else
				CIO::message(M_INFO, " done! #iter=%d\n", niter);
			break;
		}


		// hessian updates
		hstep=-hessres[maxpidx]/compute_H(maxpidx,maxpidx);
		//hstep[0]=-hessres[maxpidx]/(compute_H(maxpidx,maxpidx)+hessreg);
		//hstep[1]=-hessres[maxpidx+n]/(compute_H(maxpidx,maxpidx)+hessreg);

		hessest-=F[maxpidx]*hstep;
		//hessest[0]-=F[maxpidx]*hstep[0];
		//hessest[1]-=F[maxpidx+n]*hstep[1];

		// do primal updates ..
		DREAL tmpalpha = alphas[maxpidx] - dalphas[maxpidx]/compute_H(maxpidx,maxpidx);

		if (tmpalpha > d-alpha_eps) 
			tmpalpha = d;

		if (tmpalpha < 0+alpha_eps)
			tmpalpha = 0;

		// update alphas & dalphas & detas ...
		DREAL alphachange = tmpalpha - alphas[maxpidx];
		alphas[maxpidx] = tmpalpha;

		KERNELCACHE_ELEM* h=lock_kernel_row(maxpidx);
		for (int i=0; i<n; i++)
		{
			hessres[i]+=h[i]*hstep;
			//hessres[i]+=h[i]*hstep[0];
			//hessres[i+n]+=h[i]*hstep[1];
			dalphas[i] +=h[i]*alphachange;
		}
		unlock_kernel_row(maxpidx);

		detas+=F[maxpidx]*alphachange;
		//detas[0]+=F[maxpidx]*alphachange;
		//detas[1]+=F[maxpidx+n]*alphachange;

		// if at primal minimum, do eta update ...            
		if (primalcool)
		{
			//DREAL etachange[2] = { detas[0]/hessest[0] , detas[1]/hessest[1] };
			DREAL etachange = detas/hessest;

			etas+=etachange;        
			//etas[0]+=etachange[0];        
			//etas[1]+=etachange[1];        

			// update dalphas
			for (int i=0; i<n; i++)
				dalphas[i]+= F[i] * etachange;
				//dalphas[i]+= F[i] * etachange[0] + F[i+n] * etachange[1];
		}
	}

	if (niter >= maxiter)
		CIO::message(M_WARN, "increase maxiter ... \n");


	int nsv=0;
	for (int i=0; i<n; i++)
	{
		if (alphas[i]>0)
			nsv++;
	}


	create_new_model(nsv);
	//set_bias(etas[0]/etas[1]);
	set_bias(etas);

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
