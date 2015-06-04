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

#include <shogun/machine/gp/SingleFITCLaplacianBase.h>

#ifdef HAVE_NLOPT
#include <nlopt.h>
#include <shogun/features/DenseFeatures.h>
#endif

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DotFeatures.h>

using namespace shogun;
using namespace Eigen;

CSingleFITCLaplacianBase::CSingleFITCLaplacianBase() : CSingleSparseInferenceBase()
{
	init();
}

CSingleFITCLaplacianBase::CSingleFITCLaplacianBase(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat)
		: CSingleSparseInferenceBase(kern, feat, m, lab, mod, lat)
{
	init();
}

void CSingleFITCLaplacianBase::init()
{
	m_lock=new CLock();
	SG_ADD(&m_al, "al", "alpha", MS_NOT_AVAILABLE);
	SG_ADD(&m_t, "t", "noise", MS_NOT_AVAILABLE);
	SG_ADD(&m_B, "B", "B", MS_NOT_AVAILABLE);
	SG_ADD(&m_w, "w", "B*al", MS_NOT_AVAILABLE);
	SG_ADD(&m_Rvdd, "Rvdd", "Rvdd", MS_NOT_AVAILABLE);
	SG_ADD(&m_V, "V", "V", MS_NOT_AVAILABLE);
}

CSingleFITCLaplacianBase::~CSingleFITCLaplacianBase()
{
	delete m_lock;
}


SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_related_cov_diagonal()
{
	//time complexity O(m*n)
	Map<MatrixXd> eigen_W(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);

	SGVector<float64_t> res(m_al.vlen);
	Map<VectorXd> eigen_res(res.vector, res.vlen);
	//-sum(W.*W,1)' - al.*al;
	eigen_res=-eigen_W.cwiseProduct(eigen_W).colwise().sum().transpose()-eigen_al.array().pow(2).matrix();
	return res;
}

float64_t CSingleFITCLaplacianBase::get_derivative_related_cov_helper(
	SGMatrix<float64_t> dKuui, SGVector<float64_t> v, SGMatrix<float64_t> R)
{
	//time complexity O(m^2*n)
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_W(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

	Map<MatrixXd> eigen_dKuui(dKuui.matrix, dKuui.num_rows, dKuui.num_cols);
	Map<VectorXd> eigen_v(v.vector, v.vlen);
	Map<MatrixXd> eigen_R(R.matrix, R.num_rows, R.num_cols);

	//-al'*(v.*al)-sum(W.*W,1)*v = v'*(-sum(W.*W,1)'-(al.*al))
	SGVector<float64_t> di=get_derivative_related_cov_diagonal();
	Map<VectorXd> eigen_di(di.vector, di.vlen);

	//(w'*dKuui*w -al'*(v.*al)- sum(W.*W,1)*v - sum(sum((R*W').*BWt)))/2;
	float64_t result=(eigen_w.dot(eigen_dKuui*eigen_w)+eigen_v.dot(eigen_di)-
			(eigen_R*eigen_W.adjoint()).cwiseProduct(eigen_B*eigen_W.adjoint()).sum())/2.0;

	return result;
}

float64_t CSingleFITCLaplacianBase::get_derivative_related_cov(SGVector<float64_t> ddiagKi,
	SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui)
{
	//time complexity O(m^2*n)
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);
	Map<VectorXd> eigen_ddiagKi(ddiagKi.vector, ddiagKi.vlen);
	Map<MatrixXd> eigen_dKuui(dKuui.matrix, dKuui.num_rows, dKuui.num_cols);
	Map<MatrixXd> eigen_dKui(dKui.matrix, dKui.num_rows, dKui.num_cols);

	// compute R=2*dKui-dKuui*B
	SGMatrix<float64_t> R(dKui.num_rows, dKui.num_cols);
	Map<MatrixXd> eigen_R(R.matrix, R.num_rows, R.num_cols);
	eigen_R=2*eigen_dKui-eigen_dKuui*eigen_B;

	// compute v=ddiagKi-sum(R.*B, 1)'
	SGVector<float64_t> v(ddiagKi.vlen);
	Map<VectorXd> eigen_v(v.vector, v.vlen);
	eigen_v=eigen_ddiagKi-eigen_R.cwiseProduct(eigen_B).colwise().sum().adjoint();

	return get_derivative_related_cov(ddiagKi, dKuui, dKui, v, R);
}

float64_t CSingleFITCLaplacianBase::get_derivative_related_cov(SGVector<float64_t> ddiagKi,
	SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui,
	SGVector<float64_t> v, SGMatrix<float64_t> R)
{
	//time complexity O(m^2*n)
	Map<VectorXd> eigen_t(m_t.vector, m_t.vlen);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_W(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	Map<VectorXd> eigen_ddiagKi(ddiagKi.vector, ddiagKi.vlen);
	Map<MatrixXd> eigen_dKuui(dKuui.matrix, dKuui.num_rows, dKuui.num_cols);
	Map<MatrixXd> eigen_dKui(dKui.matrix, dKui.num_rows, dKui.num_cols);

	//(w'*dKuui*w -al'*(v.*al)- sum(W.*W,1)*v - sum(sum((R*W').*BWt)))/2;
	float64_t result=get_derivative_related_cov_helper(dKuui, v, R);

	// compute dnlZ=(ddiagKi'*(1./g_sn2)+w'*(dKuui*w-2*(dKui*al))-al'*(v.*al)-
	// sum(W.*W,1)*v- sum(sum((R*W').*(B*W'))))/2;
	result+=(eigen_ddiagKi.dot(eigen_t))/2.0-
			eigen_w.dot((eigen_dKui*eigen_al));
	return result;
}

SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	// the time complexity O(m^2*n) if the TO DO is done
	REQUIRE(param, "Param not set\n");
	REQUIRE(!(strcmp(param->m_name, "scale")
			&& strcmp(param->m_name, "inducing_noise")
			&& strcmp(param->m_name, "inducing_features")),
		    "Can't compute derivative of"
			" the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	if (!strcmp(param->m_name, "inducing_noise"))
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
			SG_WARNING("Derivative wrt %s cannot be computed since the kernel does not support fully FITC inference\n",
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
	ddiagKi*=m_scale*2.0;
	dKuui*=m_scale*2.0;
	dKui*=m_scale*2.0;

	SGVector<float64_t> result(1);

	result[0]=get_derivative_related_cov(deriv_trtr, deriv_uu, deriv_tru);
	return result;
}

SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_wrt_kernel(
		const TParameter* param)
{
	// the time complexity O(m^2*n) if the TO DO is done
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

		ddiagKi*=CMath::sq(m_scale);
		dKuui*=CMath::sq(m_scale);
		dKui*=CMath::sq(m_scale);

		result[i]=get_derivative_related_cov(deriv_trtr, deriv_uu, deriv_tru);
	}
	SG_UNREF(inducing_features);
	return result;
}

float64_t CSingleFITCLaplacianBase::get_derivative_related_mean(SGVector<float64_t> dmu)
{
	//time complexity O(n)
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);
	return -eigen_dmu.dot(eigen_al);
}

SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_wrt_mean(
		const TParameter* param)
{
	//time complexity O(n)
	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;

		dmu=m_mean->get_parameter_derivative(m_features, param, i);

		// compute dnlZ=-dm'*al
		result[i]=get_derivative_related_mean(dmu);
	}

	return result;
}

SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_wrt_inducing_noise(
	const TParameter* param)
{
	//time complexity O(m^2*n)
	REQUIRE(param, "Param not set\n");
	REQUIRE(!strcmp(param->m_name, "inducing_noise"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

	SGMatrix<float64_t> R(m_B.num_rows, m_B.num_cols);
	Map<MatrixXd> eigen_R(R.matrix, R.num_rows, R.num_cols);
	//dKuui = 2*snu2; R = -dKuui*B;
	float64_t factor=2.0*m_ind_noise;
	eigen_R=-eigen_B*factor;

	SGVector<float64_t> v(m_B.num_cols);
	Map<VectorXd> eigen_v(v.vector, v.vlen);
	//v = -sum(R.*B,1)';
	eigen_v=-eigen_R.cwiseProduct(eigen_B).colwise().sum().adjoint();

	SGMatrix<float64_t> dKuui=SGMatrix<float64_t>::create_identity_matrix(m_w.vlen,factor);

	SGVector<float64_t> result(1);
	//(w'*dKuui*w -al'*(v.*al)- sum(W.*W,1)*v - sum(sum((R*W').*BWt)))/2;
	result[0]=get_derivative_related_cov_helper(dKuui, v, R);

	return result;
}

SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_related_inducing_features(
	SGMatrix<float64_t> BdK, const TParameter* param)
{
	//time complexity depends on the implementation of the provided kernel
	//time complexity is at least O(p*n*m), where p is the dimension (#) of features
	//For an ARD kernel with KL_FULL, the time complexity is O(p*n*m*d),
	//where the paramter \f$\Lambda\f$ of the ARD kerenl is a \f$d\f$-by-\f$p\f$ matrix,
	//For an ARD kernel with KL_SCALAR and KL_DIAG, the time complexity is O(p*n*m)
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);
	Map<MatrixXd> eigen_BdK(BdK.matrix, BdK.num_rows, BdK.num_cols);

	int32_t dim=m_inducing_features.num_rows;
	int32_t num_samples=m_inducing_features.num_cols;
	SGVector<float64_t>deriv_lat(dim*num_samples);
	deriv_lat.zero();

	m_lock->lock();
	CFeatures *inducing_features=get_inducing_features();
	//asymtric part (related to xu and x)
	m_kernel->init(inducing_features, m_features);
	//A = (Kpu.*BdK)*diag(e);
	//Kpu=1 in our setting
	MatrixXd A=CMath::sq(m_scale)*eigen_BdK;
	for(int32_t lat_idx=0; lat_idx<A.rows(); lat_idx++)
	{
		Map<VectorXd> deriv_lat_col_vec(deriv_lat.vector+lat_idx*dim,dim);
		SGMatrix<float64_t> deriv_mat=m_kernel->get_parameter_gradient(param, lat_idx);
		Map<MatrixXd> eigen_deriv_mat(deriv_mat.matrix, deriv_mat.num_rows, deriv_mat.num_cols);
		deriv_lat_col_vec+=eigen_deriv_mat*(-A.row(lat_idx).transpose());
	}
	m_lock->unlock();

	m_lock->lock();
	//symtric part (related to xu and xu)
	m_kernel->init(inducing_features, inducing_features);
	//C = (Kpuu.*(BdK*B'))*diag(e);
	//Kpuu=1 in our setting
	MatrixXd C=CMath::sq(m_scale)*(eigen_BdK*eigen_B.transpose());
	for(int32_t lat_lidx=0; lat_lidx<C.rows(); lat_lidx++)
	{
		Map<VectorXd> deriv_lat_col_vec(deriv_lat.vector+lat_lidx*dim,dim);
		SGMatrix<float64_t> deriv_mat=m_kernel->get_parameter_gradient(param, lat_lidx);
		Map<MatrixXd> eigen_deriv_mat(deriv_mat.matrix, deriv_mat.num_rows, deriv_mat.num_cols);
		deriv_lat_col_vec+=eigen_deriv_mat*(C.row(lat_lidx).transpose());
	}
	SG_UNREF(inducing_features);
	m_lock->unlock();
	return deriv_lat;
}

SGVector<float64_t> CSingleFITCLaplacianBase::get_derivative_wrt_inducing_features(const TParameter* param)
{
	//time complexity depends on the implementation of the provided kernel
	//time complexity is at least O(max((p*n*m),(m^2*n))), where p is the dimension (#) of features
	//For an ARD kernel with KL_FULL, the time complexity is O(max((p*n*m*d),(m^2*n)))
	//where the paramter \f$\Lambda\f$ of the ARD kerenl is a \f$d\f$-by-\f$p\f$ matrix,
	//For an ARD kernel with KL_SCALE and KL_DIAG, the time complexity is O(max((p*n*m),(m^2*n)))
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<MatrixXd> eigen_W(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

	//v = diag_dK-1./g_sn2;
	SGVector<float64_t> v=get_derivative_related_cov_diagonal();
	Map<VectorXd> eigen_v(v.vector, v.vlen);

	//BdK = B.*repmat(v',nu,1) + BWt*W + (B*al)*al';
	SGMatrix<float64_t> BdK(m_B.num_rows, m_B.num_cols);
	Map<MatrixXd> eigen_BdK(BdK.matrix, BdK.num_rows, BdK.num_cols);
	eigen_BdK=eigen_B*eigen_v.asDiagonal()+eigen_w*(eigen_al.transpose())+
		eigen_B*eigen_W.transpose()*eigen_W;

	return get_derivative_related_inducing_features(BdK, param);
}
#endif /* HAVE_EIGEN3 */
