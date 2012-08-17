/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/evaluation/CrossValidationMKLStorage.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/classifier/mkl/MKL.h>

using namespace shogun;

void CCrossValidationMKLStorage::update_trained_machine(
		CMachine* machine, const char* prefix)
{
	if (!dynamic_cast<CMKL*>(machine))
	{
		SG_ERROR("%s::update_trained_machine(): This method is only usable "
				"with CMKL derived machines. This one is \"s\"\n", get_name(),
				machine->get_name());
	}

	CMKL* mkl=(CMKL*)machine;
	CCombinedKernel* kernel=dynamic_cast<CCombinedKernel*>(
			mkl->get_kernel());

	SGVector<float64_t> w=kernel->get_subkernel_weights();

	/* evtl allocate memory (first call) */
	if (!m_mkl_weights.matrix)
	{
		SG_PRINT("allocating memory for mkl weight matrix\n");
		m_mkl_weights=SGMatrix<float64_t>(w.vlen, m_num_folds*m_num_runs);
	}

	/* put current mkl weights into matrix, copy memory vector wise to make
	 * things fast */
	index_t n=m_current_run_index*m_current_fold_index;
	index_t first_idx=n*w.vlen+m_current_fold_index*w.vlen;
	SG_DEBUG("run %d, fold %d, matrix index %d\n",m_current_run_index,
			m_current_fold_index, first_idx);
	memcpy(&m_mkl_weights.matrix[first_idx], w.vector,
			w.vlen*sizeof(float64_t));

	SG_UNREF(kernel);
}
