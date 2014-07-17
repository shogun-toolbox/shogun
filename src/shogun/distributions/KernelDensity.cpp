/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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
 */

#include <shogun/distributions/KernelDensity.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/tree/KDTree.h>
#include <shogun/multiclass/tree/BallTree.h> 

using namespace shogun;

CKernelDensity::CKernelDensity(float64_t bandwidth, EKernelType kernel, EDistanceType dist, EEvaluationMode eval, int32_t leaf_size, float64_t atol, float64_t rtol)
: CDistribution()
{
	init();

	m_bandwidth=bandwidth;
	m_eval=eval;
	m_leaf_size=leaf_size;
	m_atol=atol;
	m_rtol=rtol;
	m_dist=dist;
	m_kernel_type=kernel;
}

CKernelDensity::~CKernelDensity()
{
	SG_UNREF(tree);
}

bool CKernelDensity::train(CFeatures* data)
{
	REQUIRE(data,"Data not supplied\n")
	CDenseFeatures<float64_t>* dense_data=CDenseFeatures<float64_t>::obtain_from_generic(data);

	SG_UNREF(tree);
	switch (m_eval)
	{
		case EM_KDTREE_SINGLE:
		{
			tree=new CKDTree(m_leaf_size,m_dist);
			break;
		}
		case EM_BALLTREE_SINGLE:
		{
			tree=new CBallTree(m_leaf_size,m_dist);
			break;
		}
		default:
		{
			SG_ERROR("Evaluation mode not recognized\n");
		}
	}

	tree->build_tree(dense_data);
	return true;
}

SGVector<float64_t> CKernelDensity::get_log_density(CDenseFeatures<float64_t>* test)
{
	REQUIRE(test,"data not supplied\n")
	return tree->log_kernel_density(test->get_feature_matrix(),m_kernel_type,m_bandwidth,m_atol,m_rtol);
}

int32_t CKernelDensity::get_num_model_parameters()
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CKernelDensity::get_log_model_parameter(int32_t num_param)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CKernelDensity::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CKernelDensity::get_log_likelihood_example(int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

void CKernelDensity::init()
{
	m_bandwidth=1.0;
	m_eval=EM_KDTREE_SINGLE;
	m_kernel_type=K_GAUSSIAN;
	m_dist=D_EUCLIDEAN;
	m_leaf_size=1;
	m_atol=0;
	m_rtol=0;
	tree=NULL;

	SG_ADD(&m_bandwidth,"m_bandwidth","bandwidth",MS_NOT_AVAILABLE);
	SG_ADD(&m_leaf_size,"m_leaf_size","leaf size",MS_NOT_AVAILABLE);
	SG_ADD(&m_atol,"m_atol","absolute tolerance",MS_NOT_AVAILABLE);
	SG_ADD(&m_rtol,"m_rtol","relative tolerance",MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &tree,"tree","tree",MS_NOT_AVAILABLE);
}