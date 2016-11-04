/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */

#include <shogun/machine/gp/SingleSparseInference.h>
#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
#include <shogun/optimization/NLOPTMinimizer.h>
#endif //HAVE_NLOPT
#endif //USE_GPL_SHOGUN

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/optimization/FirstOrderBoundConstraintsCostFunction.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** Wrapped cost function used for the NLOPT minimizer */
class SingleSparseInferenceCostFunction: public FirstOrderBoundConstraintsCostFunction
{
public:
        SingleSparseInferenceCostFunction():FirstOrderBoundConstraintsCostFunction() {  init(); }
        virtual ~SingleSparseInferenceCostFunction() { SG_UNREF(m_obj); }
	virtual const char* get_name() const { return "SingleSparseInferenceCostFunction"; }
        void set_target(CSingleSparseInference *obj)
	{
		REQUIRE(obj,"Object not set\n");
		if(obj!=m_obj)
		{
			SG_REF(obj);
			SG_UNREF(m_obj);
			m_obj=obj;
			m_obj->check_fully_sparse();
			REQUIRE(m_obj->m_fully_sparse,"Can not compute gradient\n");
		}
	}
        void unset_target(bool is_unref)
	{
		if(is_unref)
		{
			SG_UNREF(m_obj);
		}
		m_obj=NULL;
	}
        virtual float64_t get_cost()
	{
		REQUIRE(m_obj,"Object not set\n");
		float64_t nlz=m_obj->get_negative_log_marginal_likelihood();
		return nlz;
	}
        virtual SGVector<float64_t> obtain_variable_reference()
	{
		REQUIRE(m_obj,"Object not set\n");
		SGMatrix<float64_t>& lat_m=m_obj->m_inducing_features;
		SGVector<float64_t> x(lat_m.matrix,lat_m.num_rows*lat_m.num_cols,false);
		return x;
	}
        virtual SGVector<float64_t> get_gradient()
	{
		REQUIRE(m_obj,"Object not set\n");
		m_obj->compute_gradient();
		TParameter* param=m_obj->m_gradient_parameters->get_parameter("inducing_features");
		SGVector<float64_t> derivatives=m_obj->get_derivative_wrt_inducing_features(param);
		return derivatives;
	}
        virtual SGVector<float64_t> get_lower_bound()
	{
		REQUIRE(m_obj,"Object not set\n");
		return m_obj->m_lower_bound;
	}
        virtual SGVector<float64_t> get_upper_bound()
	{
		REQUIRE(m_obj,"Object not set\n");
		return m_obj->m_upper_bound;
	}
private:
        CSingleSparseInference *m_obj;
        void init()
	{
		m_obj=NULL;
		//The existing implementation in CSGObject::get_parameter_incremental_hash()
		//can NOT deal with circular reference when parameter_hash_changed() is called
		//SG_ADD((CSGObject **)&m_obj, "CSigleSparseInference__m_obj",
			//"m_obj in SingleSparseInferenceCostFunction", MS_NOT_AVAILABLE);
	}
};
#endif //DOXYGEN_SHOULD_SKIP_THIS

CSingleSparseInference::CSingleSparseInference() : CSparseInference()
{
	init();
}

CSingleSparseInference::CSingleSparseInference(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat)
		: CSparseInference(kern, feat, m, lab, mod, lat)
{
	init();
	check_fully_sparse();
}

void CSingleSparseInference::init()
{
	m_fully_sparse=false;
	m_inducing_minimizer=NULL;
	SG_ADD(&m_fully_sparse, "fully_Sparse",
		"whether the kernel support sparse inference", MS_NOT_AVAILABLE);
	m_lock=new CLock();

	SG_ADD(&m_upper_bound, "upper_bound",
		"upper bound of inducing features", MS_NOT_AVAILABLE);
	SG_ADD(&m_lower_bound, "lower_bound",
		"lower bound of inducing features", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_ind_iterations, "max_ind_iterations",
		"max number of iterations used in inducing features optimization", MS_NOT_AVAILABLE);
	SG_ADD(&m_ind_tolerance, "ind_tolerance",
		"tolearance used in inducing features optimization", MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_inducing_features,
		"opt_inducing_features", "whether optimize inducing features", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject **)&m_inducing_minimizer,
		"inducing_minimizer", "Minimizer used in optimize inducing features", MS_NOT_AVAILABLE);

	m_max_ind_iterations=50;
	m_ind_tolerance=1e-3;
	m_opt_inducing_features=false;
	m_lower_bound=SGVector<float64_t>();
	m_upper_bound=SGVector<float64_t>();
}

void CSingleSparseInference::set_kernel(CKernel* kern)
{
	CInference::set_kernel(kern);
	check_fully_sparse();
}

CSingleSparseInference::~CSingleSparseInference()
{
	SG_UNREF(m_inducing_minimizer);
	delete m_lock;
}

void CSingleSparseInference::check_fully_sparse()
{
	REQUIRE(m_kernel, "Kernel must be set first\n")
	if (strstr(m_kernel->get_name(), "SparseKernel")!=NULL)
		m_fully_sparse=true;
	else
	{
		SG_WARNING( "The provided kernel does not support to optimize inducing features\n");
		m_fully_sparse=false;
	}
}

SGVector<float64_t> CSingleSparseInference::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	// the time complexity O(m^2*n) if the TO DO is done
	REQUIRE(param, "Param not set\n");
	REQUIRE(!(strcmp(param->m_name, "log_scale")
			&& strcmp(param->m_name, "log_inducing_noise")
			&& strcmp(param->m_name, "inducing_features")),
		    "Can't compute derivative of"
			" the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	if (!strcmp(param->m_name, "log_inducing_noise"))
		// wrt inducing_noise
		// compute derivative wrt inducing noise
		return get_derivative_wrt_inducing_noise(param);
	else if (!strcmp(param->m_name, "inducing_features"))
	{
		SGVector<float64_t> res;
		if (!m_fully_sparse)
		{
			int32_t dim=m_inducing_features.num_rows;
			int32_t num_samples=m_inducing_features.num_cols;
			res=SGVector<float64_t>(dim*num_samples);
			SG_WARNING("Derivative wrt %s cannot be computed since the kernel does not support fully sparse inference\n",
				param->m_name);
			res.zero();
			return res;
		}
		res=get_derivative_wrt_inducing_features(param);
		return res;
	}

	// wrt scale
	// clone kernel matrices
	SGVector<float64_t> deriv_trtr=m_ktrtr_diag.clone();
	SGMatrix<float64_t> deriv_uu=m_kuu.clone();
	SGMatrix<float64_t> deriv_tru=m_ktru.clone();

	// create eigen representation of kernel matrices
	Map<VectorXd> ddiagKi(deriv_trtr.vector, deriv_trtr.vlen);
	Map<MatrixXd> dKuui(deriv_uu.matrix, deriv_uu.num_rows, deriv_uu.num_cols);
	Map<MatrixXd> dKui(deriv_tru.matrix, deriv_tru.num_rows, deriv_tru.num_cols);

	// compute derivatives wrt scale for each kernel matrix
	SGVector<float64_t> result(1);

	result[0]=get_derivative_related_cov(deriv_trtr, deriv_uu, deriv_tru);
	result[0]*=CMath::exp(m_log_scale*2.0)*2.0;
	return result;
}

SGVector<float64_t> CSingleSparseInference::get_derivative_wrt_kernel(
	const TParameter* param)
{
	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	CFeatures *inducing_features=get_inducing_features();
	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> deriv_trtr;
		SGMatrix<float64_t> deriv_uu;
		SGMatrix<float64_t> deriv_tru;

		m_lock->lock();
		m_kernel->init(m_features, m_features);
		//to reduce the time complexity
		//the kernel object only computes diagonal elements of gradients wrt hyper-parameter
		deriv_trtr=m_kernel->get_parameter_gradient_diagonal(param, i);

		m_kernel->init(inducing_features, inducing_features);
		deriv_uu=m_kernel->get_parameter_gradient(param, i);

		m_kernel->init(inducing_features, m_features);
		deriv_tru=m_kernel->get_parameter_gradient(param, i);
		m_lock->unlock();

		// create eigen representation of derivatives
		Map<VectorXd> ddiagKi(deriv_trtr.vector, deriv_trtr.vlen);
		Map<MatrixXd> dKuui(deriv_uu.matrix, deriv_uu.num_rows,
				deriv_uu.num_cols);
		Map<MatrixXd> dKui(deriv_tru.matrix, deriv_tru.num_rows,
				deriv_tru.num_cols);

		result[i]=get_derivative_related_cov(deriv_trtr, deriv_uu, deriv_tru);
		result[i]*=CMath::exp(m_log_scale*2.0);
	}
	SG_UNREF(inducing_features);
	return result;
}

void CSingleSparseInference::check_bound(SGVector<float64_t> bound, const char* name)
{
	if (bound.vlen>1)
	{
		REQUIRE(m_inducing_features.num_rows>0, "Inducing features must set before this method is called\n");
		REQUIRE(m_inducing_features.num_rows*m_inducing_features.num_cols==bound.vlen,
			"The length of inducing features (%dx%d)",
			" and the length of bound constraints (%d) are different\n", 
			m_inducing_features.num_rows,m_inducing_features.num_cols,bound.vlen);
	}
	else if(bound.vlen==1)
	{
		SG_WARNING("All inducing_features (%dx%d) are constrainted by the single value (%f) in the %s bound\n",
			m_inducing_features.num_rows,m_inducing_features.num_cols,bound[0],name);
	}
}

void CSingleSparseInference::set_lower_bound_of_inducing_features(SGVector<float64_t> bound)
{
	check_bound(bound,"lower");
	m_lower_bound=bound;
}
void CSingleSparseInference::set_upper_bound_of_inducing_features(SGVector<float64_t> bound)
{
	check_bound(bound, "upper");
	m_upper_bound=bound;
}

void CSingleSparseInference::set_max_iterations_for_inducing_features(int32_t it)
{
	REQUIRE(it>0, "Iteration (%d) must be positive\n",it);
	m_max_ind_iterations=it;
}
void CSingleSparseInference::set_tolearance_for_inducing_features(float64_t tol)
{

	REQUIRE(tol>0, "Tolearance (%f) must be positive\n",tol);
	m_ind_tolerance=tol;
}
void CSingleSparseInference::enable_optimizing_inducing_features(bool is_optmization, FirstOrderMinimizer* minimizer)
{
	m_opt_inducing_features=is_optmization;
	if (m_opt_inducing_features)
	{
		check_fully_sparse();
		REQUIRE(m_fully_sparse,"Please use a kernel which has the functionality about optimizing inducing features\n");
	}
	if(minimizer)
	{
		if (minimizer!=m_inducing_minimizer)
		{
			SG_REF(minimizer);
			SG_UNREF(m_inducing_minimizer);
			m_inducing_minimizer=minimizer;
		}
	}
	else
	{

		SG_UNREF(m_inducing_minimizer);
#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
		m_inducing_minimizer=new CNLOPTMinimizer();
		SG_REF(m_inducing_minimizer);
#else
		m_inducing_minimizer=NULL;
		SG_WARNING("We require NLOPT library for using default minimizer.\nYou can use other minimizer. (eg, LBFGSMinimier)\n");
#endif //HAVE_NLOPT
#else 
		m_inducing_minimizer=NULL;
		SG_WARNING("We require NLOPT (GPL License) library for using default minimizer.\nYou can use other minimizer. (eg, LBFGSMinimier)");
#endif //USE_GPL_SHOGUN
	}
}

void CSingleSparseInference::optimize_inducing_features()
{
	if (!m_opt_inducing_features)
		return;

	REQUIRE(m_inducing_minimizer, "Please call enable_optimizing_inducing_features() first\n");
	SingleSparseInferenceCostFunction *cost_fun=new SingleSparseInferenceCostFunction();
	cost_fun->set_target(this);
	bool cleanup=false;
#ifdef USE_REFERENCE_COUNTING
	if(this->ref_count()>1)
		cleanup=true;
#endif

#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
	CNLOPTMinimizer* opt=dynamic_cast<CNLOPTMinimizer*>(m_inducing_minimizer);
	if (opt)
		opt->set_nlopt_parameters(LD_LBFGS, m_max_ind_iterations, m_ind_tolerance, m_ind_tolerance);
#endif //HAVE_NLOPT
#endif //USE_GPL_SHOGUN

	m_inducing_minimizer->set_cost_function(cost_fun);
	m_inducing_minimizer->minimize();
	m_inducing_minimizer->unset_cost_function(false);
	cost_fun->unset_target(cleanup);
	SG_UNREF(cost_fun);
}

}
