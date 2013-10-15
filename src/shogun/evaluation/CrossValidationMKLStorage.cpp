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
#include <shogun/classifier/mkl/MKLMulticlass.h>

using namespace shogun;

void CCrossValidationMKLStorage::update_trained_machine(
		CMachine* machine, const char* prefix)
{
	REQUIRE(machine, "%s::update_trained_machine(): Provided Machine is NULL!\n",
			get_name());

	CMKL* mkl=dynamic_cast<CMKL*>(machine);
	CMKLMulticlass* mkl_multiclass=dynamic_cast<CMKLMulticlass*>(machine);
	REQUIRE(mkl || mkl_multiclass, "%s::update_trained_machine(): This method is only usable "
				"with CMKL derived machines. This one is \"%s\"\n", get_name(),
				machine->get_name());

	CKernel* kernel = NULL;
	if (mkl)
		kernel = mkl->get_kernel();
	else
		kernel = mkl_multiclass->get_kernel();

	REQUIRE(kernel, "%s::update_trained_machine(): No kernel assigned to "
			"machine of type \"%s\"\n", get_name(), machine->get_name());

	CCombinedKernel* combined_kernel=dynamic_cast<CCombinedKernel*>(kernel);
	REQUIRE(combined_kernel, "%s::update_trained_machine(): This method is only"
			" usable with CCombinedKernel on machines. This one is \"s\"\n",
			get_name(), kernel->get_name());

	SGVector<float64_t> w=combined_kernel->get_subkernel_weights();

	/* evtl re-allocate memory (different number of runs from evaluation before) */
	if (m_mkl_weights.num_rows!=w.vlen ||
			m_mkl_weights.num_cols!=m_num_folds*m_num_runs)
	{
		if (m_mkl_weights.matrix)
		{
			SG_DEBUG("deleting memory for mkl weight matrix\n")
			m_mkl_weights=SGMatrix<float64_t>();
		}
	}

	/* evtl allocate memory (first call) */
	if (!m_mkl_weights.matrix)
	{
		SG_DEBUG("allocating memory for mkl weight matrix\n")
		m_mkl_weights=SGMatrix<float64_t>(w.vlen,m_num_folds*m_num_runs);
	}

	/* put current mkl weights into matrix, copy memory vector wise to make
	 * things fast. Compute index of address to where vector goes */

	/* number of runs is w.vlen*m_num_folds shift */
	index_t run_shift=m_current_run_index*w.vlen*m_num_folds;

	/* fold shift is m_current_fold_index*w-vlen */
	index_t fold_shift=m_current_fold_index*w.vlen;

	/* add both index shifts */
	index_t first_idx=run_shift+fold_shift;
	SG_DEBUG("run %d, fold %d, matrix index %d\n",m_current_run_index,
			m_current_fold_index, first_idx);

	/* copy memory */
	memcpy(&m_mkl_weights.matrix[first_idx], w.vector,
			w.vlen*sizeof(float64_t));

	SG_UNREF(kernel);
}
