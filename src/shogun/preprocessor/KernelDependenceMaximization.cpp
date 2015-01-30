/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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
 */

#include <shogun/kernel/CustomKernel.h>
#include <shogun/statistics/KernelIndependenceTest.h>
#include <shogun/preprocessor/KernelDependenceMaximization.h>

using namespace shogun;

CKernelDependenceMaximization::CKernelDependenceMaximization()
	: CDependenceMaximization()
{
	initialize();
}

void CKernelDependenceMaximization::initialize()
{
	SG_ADD((CSGObject**)&m_kernel_features, "kernel_features",
			"the kernel to be used for features", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_kernel_labels, "kernel_labels",
			"the kernel to be used for labels", MS_NOT_AVAILABLE);

	m_kernel_features=NULL;
	m_kernel_labels=NULL;
}

CKernelDependenceMaximization::~CKernelDependenceMaximization()
{
	SG_UNREF(m_kernel_features);
	SG_UNREF(m_kernel_labels);
}

void CKernelDependenceMaximization::precompute()
{
	SG_DEBUG("Entering!\n");

	REQUIRE(m_labels_feats, "Features for labels is not initialized!\n");
	REQUIRE(m_kernel_labels, "Kernel for labels is not initialized!\n");

	// ASSERT here because the estimator is set internally and cannot
	// be set via public API
	ASSERT(m_estimator);

	CFeatureSelection<float64_t>::precompute();

	// make sure that we have an instance of CKernelIndependenceTest via
	// proper cast and set this kernel to the estimator
	CKernelIndependenceTest* estimator
		=dynamic_cast<CKernelIndependenceTest*>(m_estimator);
	ASSERT(estimator);

	// precompute the kernel for labels
	m_kernel_labels->init(m_labels_feats, m_labels_feats);
	CCustomKernel* precomputed
		=new CCustomKernel(m_kernel_labels->get_kernel_matrix());

	// replace the kernel for labels with precomputed kernel
	SG_UNREF(m_kernel_labels);
	m_kernel_labels=precomputed;
	SG_REF(m_kernel_labels);

	// we can safely SG_UNREF the feature object for labels now
	SG_UNREF(m_labels_feats);
	m_labels_feats=NULL;

	// finally set this as kernel for the labels
	estimator->set_kernel_q(m_kernel_labels);

	SG_DEBUG("Leaving!\n");
}

void CKernelDependenceMaximization::set_kernel_features(CKernel* kernel)
{
	// sanity check. using assert here because estimator instances are
	// set internally and cannot be set via public API.
	ASSERT(m_estimator);
	CKernelIndependenceTest* estimator
		=dynamic_cast<CKernelIndependenceTest*>(m_estimator);
	ASSERT(estimator);

	SG_REF(kernel);
	SG_UNREF(m_kernel_features);
	m_kernel_features=kernel;

	estimator->set_kernel_p(m_kernel_features);
}

void CKernelDependenceMaximization::set_kernel_labels(CKernel* kernel)
{
	// sanity check. using assert here because estimator instances are
	// set internally and cannot be set via public API.
	ASSERT(m_estimator);
	CKernelIndependenceTest* estimator
		=dynamic_cast<CKernelIndependenceTest*>(m_estimator);
	ASSERT(estimator);

	SG_REF(kernel);
	SG_UNREF(m_kernel_labels);
	m_kernel_labels=kernel;

	estimator->set_kernel_q(m_kernel_labels);
}

CKernel* CKernelDependenceMaximization::get_kernel_features() const
{
	SG_REF(m_kernel_features);
	return m_kernel_features;
}

CKernel* CKernelDependenceMaximization::get_kernel_labels() const
{
	SG_REF(m_kernel_labels);
	return m_kernel_labels;
}
