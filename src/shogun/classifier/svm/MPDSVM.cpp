/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Heiko Strathmann,
 *          Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/base/progress.h>
#include <shogun/classifier/svm/MPDSVM.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

MPDSVM::MPDSVM()
: SVM()
{
}

MPDSVM::MPDSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: SVM(C, k, lab)
{
}

MPDSVM::~MPDSVM()
{
}

bool MPDSVM::train_machine(std::shared_ptr<Features> data)
{
	auto labels = binary_labels(m_labels);

	ASSERT(kernel)

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			error("Number of training vectors does not match number of labels");
		kernel->init(data, data);
	}
	ASSERT(kernel->has_features())

	//const float64_t nu=0.32;
	const float64_t alpha_eps=1e-12;
	const float64_t eps=get_epsilon();
	const int64_t maxiter = 1L<<30;
	//const bool nustop=false;
	//const int32_t k=2;
	const int32_t n = labels->get_num_labels();
	ASSERT(n>0)
	//const float64_t d = 1.0/n/nu; //NUSVC
	const float64_t d = get_C1(); //CSVC
	const float64_t primaleps=eps;
	const float64_t dualeps=eps*n; //heuristic
	int64_t niter=0;

	kernel_cache = std::make_shared<Cache<KERNELCACHE_ELEM>>(kernel->get_cache_size(), n, n);
	float64_t* alphas=SG_MALLOC(float64_t, n);
	float64_t* dalphas=SG_MALLOC(float64_t, n);
	//float64_t* hessres=SG_MALLOC(float64_t, 2*n);
	float64_t* hessres=SG_MALLOC(float64_t, n);
	//float64_t* F=SG_MALLOC(float64_t, 2*n);
	float64_t* F=SG_MALLOC(float64_t, n);

	//float64_t hessest[2]={0,0};
	//float64_t hstep[2];
	//float64_t etas[2]={0,0};
	//float64_t detas[2]={0,1}; //NUSVC
	float64_t etas=0;
	float64_t detas=0;   //CSVC
	float64_t hessest=0;
	float64_t hstep;

	const float64_t stopfac = 1;

	bool primalcool;
	bool dualcool;

	//if (nustop)
	//etas[1] = 1;

	for (int32_t i=0; i<n; i++)
	{
		alphas[i]=0;
		F[i] = labels->get_label(i);
		//F[i+n]=-1;
		hessres[i] = labels->get_label(i);
		//hessres[i+n]=-1;
		//dalphas[i]=F[i+n]*etas[1]; //NUSVC
		dalphas[i]=-1; //CSVC
	}

	auto pb = SG_PROGRESS(range(maxiter));
	// go ...
	while (niter++ < maxiter)
	{
		COMPUTATION_CONTROLLERS
		int32_t maxpidx=-1;
		float64_t maxpviol = -1;
		//float64_t maxdviol = Math::abs(detas[0]);
		float64_t maxdviol = Math::abs(detas);
		bool free_alpha=false;

		//if (Math::abs(detas[1])> maxdviol)
		//maxdviol=Math::abs(detas[1]);

		// compute kkt violations with correct sign ...
		for (int32_t i=0; i<n; i++)
		{
			float64_t v=Math::abs(dalphas[i]);

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
			error("no violation no convergence, should not happen!");

		// ... and evaluate stopping conditions
		//if (nustop)
		//stopfac = Math::max(etas[1], 1e-10);
		//else
		//stopfac = 1;

		if (niter%10000 == 0)
		{
			float64_t obj=0;

			for (int32_t i=0; i<n; i++)
			{
				obj-=alphas[i];
				for (int32_t j=0; j<n; j++)
					obj += 0.5 * labels->get_label(i) * labels->get_label(j) *
					       alphas[i] * alphas[j] * kernel->kernel(i, j);
			}

			SG_DEBUG("obj:{} pviol:{} dviol:{} maxpidx:{} iter:{}", obj, maxpviol, maxdviol, maxpidx, niter)
		}

		//for (int32_t i=0; i<n; i++)
		//	SG_DEBUG("alphas:{} dalphas:{}", alphas[i], dalphas[i])

		primalcool = (maxpviol < primaleps*stopfac);
		dualcool = (maxdviol < dualeps*stopfac) || (!free_alpha);

		// done?
		if (primalcool && dualcool)
		{
			if (!free_alpha)
				io::info(" no free alpha, stopping! #iter={}", niter);
			else
				io::info(" done! #iter={}", niter);
			break;
		}


		ASSERT(maxpidx>=0 && maxpidx<n)
		// hessian updates
		hstep=-hessres[maxpidx]/compute_H(maxpidx,maxpidx);
		//hstep[0]=-hessres[maxpidx]/(compute_H(maxpidx,maxpidx)+hessreg);
		//hstep[1]=-hessres[maxpidx+n]/(compute_H(maxpidx,maxpidx)+hessreg);

		hessest-=F[maxpidx]*hstep;
		//hessest[0]-=F[maxpidx]*hstep[0];
		//hessest[1]-=F[maxpidx+n]*hstep[1];

		// do primal updates ..
		float64_t tmpalpha = alphas[maxpidx] - dalphas[maxpidx]/compute_H(maxpidx,maxpidx);

		if (tmpalpha > d-alpha_eps)
			tmpalpha = d;

		if (tmpalpha < 0+alpha_eps)
			tmpalpha = 0;

		// update alphas & dalphas & detas ...
		float64_t alphachange = tmpalpha - alphas[maxpidx];
		alphas[maxpidx] = tmpalpha;

		KERNELCACHE_ELEM* h=lock_kernel_row(maxpidx);
		for (int32_t i=0; i<n; i++)
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
			//float64_t etachange[2] = { detas[0]/hessest[0] , detas[1]/hessest[1] };
			float64_t etachange = detas/hessest;

			etas+=etachange;
			//etas[0]+=etachange[0];
			//etas[1]+=etachange[1];

			// update dalphas
			for (int32_t i=0; i<n; i++)
				dalphas[i]+= F[i] * etachange;
			//dalphas[i]+= F[i] * etachange[0] + F[i+n] * etachange[1];
		}
		pb.print_progress();
	}

	pb.complete();
	if (niter >= maxiter)
		io::warn("increase maxiter ... ");


	int32_t nsv=0;
	for (int32_t i=0; i<n; i++)
	{
		if (alphas[i]>0)
			nsv++;
	}


	create_new_model(nsv);
	//set_bias(etas[0]/etas[1]);
	set_bias(etas);

	int32_t j=0;
	for (int32_t i=0; i<n; i++)
	{
		if (alphas[i]>0)
		{
			//set_alpha(j, alphas[i]*labels->get_label(i)/etas[1]);
			set_alpha(j, alphas[i] * labels->get_label(i));
			set_support_vector(j, i);
			j++;
		}
	}
	compute_svm_dual_objective();
	io::info("obj = {:.16f}, rho = {:.16f}",get_objective(),get_bias());
	io::info("Number of SV: {}", get_num_support_vectors());

	SG_FREE(alphas);
	SG_FREE(dalphas);
	SG_FREE(hessres);
	SG_FREE(F);

	return true;
}
