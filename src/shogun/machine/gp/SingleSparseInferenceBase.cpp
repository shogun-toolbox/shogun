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

#include <shogun/machine/gp/SingleSparseInferenceBase.h>

#ifdef HAVE_NLOPT
#include <nlopt.h>
#include <shogun/features/DenseFeatures.h>
#endif


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DotFeatures.h>

using namespace shogun;
using namespace Eigen;

CSingleSparseInferenceBase::CSingleSparseInferenceBase() : CSparseInferenceBase()
{
	init();
}

CSingleSparseInferenceBase::CSingleSparseInferenceBase(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat)
		: CSparseInferenceBase(kern, feat, m, lab, mod, lat)
{
	init();
	check_fully_sparse();
}

void CSingleSparseInferenceBase::init()
{
	m_fully_sparse=false;
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
	m_max_ind_iterations=50;
	m_ind_tolerance=1e-3;
	m_opt_inducing_features=false;
	m_lower_bound=SGVector<float64_t>();
	m_upper_bound=SGVector<float64_t>();
}

void CSingleSparseInferenceBase::set_kernel(CKernel* kern)
{
	CInferenceMethod::set_kernel(kern);
	check_fully_sparse();
}

CSingleSparseInferenceBase::~CSingleSparseInferenceBase()
{
	delete m_lock;
}

void CSingleSparseInferenceBase::check_fully_sparse()
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

SGVector<float64_t> CSingleSparseInferenceBase::get_derivative_wrt_inference_method(
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

SGVector<float64_t> CSingleSparseInferenceBase::get_derivative_wrt_kernel(
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

void CSingleSparseInferenceBase::check_bound(SGVector<float64_t> bound, const char* name)
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

void CSingleSparseInferenceBase::set_lower_bound_of_inducing_features(SGVector<float64_t> bound)
{
	check_bound(bound,"lower");
	m_lower_bound=bound;
}
void CSingleSparseInferenceBase::set_upper_bound_of_inducing_features(SGVector<float64_t> bound)
{
	check_bound(bound, "upper");
	m_upper_bound=bound;
}

void CSingleSparseInferenceBase::set_max_iterations_for_inducing_features(int32_t it)
{
	REQUIRE(it>0, "Iteration (%d) must be positive\n",it);
	m_max_ind_iterations=it;
}
void CSingleSparseInferenceBase::set_tolearance_for_inducing_features(float64_t tol)
{

	REQUIRE(tol>0, "Tolearance (%f) must be positive\n",tol);
	m_ind_tolerance=tol;
}
double CSingleSparseInferenceBase::nlopt_function(unsigned n, const double* x, double* grad, void* func_data)
{
	CSingleSparseInferenceBase* object=static_cast<CSingleSparseInferenceBase *>(func_data);
	REQUIRE(object,"func_data must be SingleSparseInferenceBase pointer\n");

	double nlz=object->get_negative_log_marginal_likelihood();
	object->compute_gradient();

	TParameter* param=object->m_gradient_parameters->get_parameter("inducing_features");
	SGVector<float64_t> derivatives=object->get_derivative_wrt_inducing_features(param);

	std::copy(derivatives.vector,derivatives.vector+n,grad);

	return nlz;
}

void CSingleSparseInferenceBase::enable_optimizing_inducing_features(bool is_optmization)
{
	m_opt_inducing_features=is_optmization;
}

void CSingleSparseInferenceBase::optimize_inducing_features()
{
#ifdef HAVE_NLOPT
	if (!m_opt_inducing_features)
		return;

	check_fully_sparse();
	REQUIRE(m_fully_sparse,"Please use a kernel which supports to optimize inducing features\n");

	//features by samples
	SGMatrix<float64_t>& lat_m=m_inducing_features;
	SGVector<double> x(lat_m.matrix,lat_m.num_rows*lat_m.num_cols,false);

	// create nlopt object and choose LBFGS
	// optimization algorithm
	nlopt_opt opt=nlopt_create(NLOPT_LD_LBFGS, lat_m.num_rows*lat_m.num_cols);

	if (m_lower_bound.vlen>0)
	{
		if(m_lower_bound.vlen==1)
			nlopt_set_lower_bounds1(opt, m_lower_bound[0]);
		else
		{
			SGVector<double> lower_bound(lat_m.num_rows*lat_m.num_cols);
			for(index_t j=0; j<lat_m.num_cols; j++)
				std::copy(m_lower_bound.vector, m_lower_bound.vector+m_lower_bound.vlen,
					lower_bound.vector+j*lat_m.num_rows);
			// set lower bound
			nlopt_set_lower_bounds(opt, lower_bound.vector);
		}
	}
	if (m_upper_bound.vlen>0)
	{
		if(m_upper_bound.vlen==1)
			nlopt_set_upper_bounds1(opt, m_upper_bound[0]);
		else
		{
			SGVector<double> upper_bound(lat_m.num_rows*lat_m.num_cols);
			for(index_t j=0; j<lat_m.num_cols; j++)
				std::copy(m_upper_bound.vector, m_upper_bound.vector+m_upper_bound.vlen,
					upper_bound.vector+j*lat_m.num_rows);
			// set upper bound
			nlopt_set_upper_bounds(opt, upper_bound.vector);
		}
	}

	// set maximum number of evaluations
	nlopt_set_maxeval(opt, m_max_ind_iterations);
	// set absolute argument tolearance
	nlopt_set_xtol_abs1(opt, m_ind_tolerance);
	nlopt_set_ftol_abs(opt, m_ind_tolerance);

	nlopt_set_min_objective(opt, CSingleSparseInferenceBase::nlopt_function, this);

	// the minimum objective value, upon return
	double minf;

	// optimize our function
	nlopt_result result=nlopt_optimize(opt, x.vector, &minf);
	REQUIRE(result>0, "NLopt failed while optimizing objective function!\n");

	// clean up
	nlopt_destroy(opt);
#else
	SG_PRINT("For this functionality we require NLOPT library\n");
#endif
}

