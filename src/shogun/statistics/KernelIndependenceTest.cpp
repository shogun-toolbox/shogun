/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
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

#include <shogun/statistics/KernelIndependenceTest.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>

using namespace shogun;

CKernelIndependenceTest::CKernelIndependenceTest() :
		CIndependenceTest()
{
	init();
}

CKernelIndependenceTest::CKernelIndependenceTest(CKernel* kernel_p,
		CKernel* kernel_q, CFeatures* p, CFeatures* q) :
		CIndependenceTest(p, q)
{
	init();

	m_kernel_p=kernel_p;
	SG_REF(kernel_p);

	m_kernel_q=kernel_q;
	SG_REF(kernel_q);
}

CKernelIndependenceTest::~CKernelIndependenceTest()
{
	SG_UNREF(m_kernel_p);
	SG_UNREF(m_kernel_q);
}

void CKernelIndependenceTest::init()
{
	SG_ADD((CSGObject**)&m_kernel_p, "kernel_p", "Kernel for samples from p",
			MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_kernel_q, "kernel_q", "Kernel for samples from q",
			MS_AVAILABLE);

	m_kernel_p=NULL;
	m_kernel_q=NULL;
}

SGVector<float64_t> CKernelIndependenceTest::sample_null()
{
	SG_DEBUG("entering!\n")

	/* compute sample statistics for null distribution */
	SGVector<float64_t> results;

	/* only do something if a custom kernel is used: use the power of pre-
	 * computed kernel matrices
	 */
	if (m_kernel_p->get_kernel_type()==K_CUSTOM &&
			m_kernel_q->get_kernel_type()==K_CUSTOM)
	{
		/* allocate memory */
		results=SGVector<float64_t>(m_num_null_samples);

		/* memory for index permutations */
		SGVector<index_t> ind_permutation(m_p->get_num_vectors());
		ind_permutation.range_fill();

		/* check if kernel is a custom kernel. In that case, changing features is
		 * not what we want but just subsetting the kernel itself */
		CCustomKernel* custom_kernel_p=(CCustomKernel*)m_kernel_p;

		for (index_t i=0; i<m_num_null_samples; ++i)
		{
			/* idea: shuffle samples from p while keeping samples from q intact
			 * and compute statistic. This is done using subsets here. add to
			 * custom kernel since it has no features to subset. CustomKernel
			 * has not to be re-initialised after each subset setting */
			SGVector<index_t>::permute_vector(ind_permutation);

			custom_kernel_p->add_row_subset(ind_permutation);
			custom_kernel_p->add_col_subset(ind_permutation);

			/* compute statistic for this permutation of mixed samples */
			results[i]=compute_statistic();

			/* remove subsets */
			custom_kernel_p->remove_row_subset();
			custom_kernel_p->remove_col_subset();
		}
	}
	else
	{
		/* in this case, just use superclass method */
		results=CIndependenceTest::sample_null();
	}


	SG_DEBUG("leaving!\n")
	return results;
}

void CKernelIndependenceTest::set_kernel_p(CKernel* kernel_p)
{
	/* ref before unref to avoid problems when instances are equal */
	SG_REF(kernel_p);
	SG_UNREF(m_kernel_p);
	m_kernel_p=kernel_p;
}

void CKernelIndependenceTest::set_kernel_q(CKernel* kernel_q)
{
	/* ref before unref to avoid problems when instances are equal */
	SG_REF(kernel_q);
	SG_UNREF(m_kernel_q);
	m_kernel_q=kernel_q;
}

CKernel* CKernelIndependenceTest::get_kernel_p()
{
	SG_REF(m_kernel_p);
	return m_kernel_p;
}

CKernel* CKernelIndependenceTest::get_kernel_q()
{
	SG_REF(m_kernel_q);
	return m_kernel_q;
}

SGMatrix<float64_t> CKernelIndependenceTest::get_kernel_matrix_K()
{
	SG_DEBUG("entering!\n");

	SGMatrix<float64_t> K;

	/* distinguish between custom and normal kernels */
	if (m_kernel_p->get_kernel_type()==K_CUSTOM)
	{
		/* custom kernels need to to be initialised when a subset is added */
		CCustomKernel* custom_kernel_p=(CCustomKernel*)m_kernel_p;
		K=custom_kernel_p->get_kernel_matrix();
	}
	else
	{
		/* need to init the kernel if kernel is not precomputed - if subsets of
		 * features are in the stack (for permutation), this will handle it */
		m_kernel_p->init(m_p, m_p);
		K=m_kernel_p->get_kernel_matrix();
	}

	SG_DEBUG("leaving!\n");

	return K;
}

SGMatrix<float64_t> CKernelIndependenceTest::get_kernel_matrix_L()
{
	SG_DEBUG("entering!\n");

	SGMatrix<float64_t> L;

	/* now second half of data for L */
	if (m_kernel_q->get_kernel_type()==K_CUSTOM)
	{
		/* custom kernels need to to be initialised - no subsets here */
		CCustomKernel* custom_kernel_q=(CCustomKernel*)m_kernel_q;
		L=custom_kernel_q->get_kernel_matrix();
	}
	else
	{
		/* need to init the kernel if kernel is not precomputed */
		m_kernel_q->init(m_q, m_q);
		L=m_kernel_q->get_kernel_matrix();
	}

	SG_DEBUG("leaving!\n");

	return L;
}

