/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
*/

/**
 * This code was ported from the previous CrossValidationMKLStorage,
 * written by Heiko Strathmann and Sergey Lisitsyn.
 */

#include <shogun/classifier/mkl/MKL.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/lib/parameter_observers/ParameterObserverCVMKL.h>

using namespace shogun;

ParameterObserverCVMKL::ParameterObserverCVMKL()
{
}

ParameterObserverCVMKL::~ParameterObserverCVMKL()
{
}

SGMatrix<float64_t> ParameterObserverCVMKL::get_mkl_weights()
{
	if (m_mkl_weights.size() == 0)
		generate_mkl_weights();
	return m_mkl_weights;
}

void ParameterObserverCVMKL::clear()
{
	for (auto o : m_observations)
		SG_UNREF(o);
	m_observations.clear();
	m_mkl_weights = SGMatrix<float64_t>();
}

void ParameterObserverCVMKL::generate_mkl_weights()
{
	for (auto obs : m_observations)
	{
		for (auto fold : obs->get_folds_results())
		{
			REQUIRE(
			    fold->get_trained_machine(),
			    "%s::generate_mkl_weights(): Provided Machine is NULL!\n",
			    get_name());

			CMKL* mkl = dynamic_cast<CMKL*>(fold->get_trained_machine());
			CMKLMulticlass* mkl_multiclass =
			    dynamic_cast<CMKLMulticlass*>(fold->get_trained_machine());
			REQUIRE(
			    mkl || mkl_multiclass,
			    "%s::set_trained_machine(): This method is only usable "
			    "with CMKL derived machines. This one is \"%s\"\n",
			    get_name(), fold->get_trained_machine()->get_name());

			CKernel* kernel = NULL;
			if (mkl)
				kernel = mkl->get_kernel();
			else
				kernel = mkl_multiclass->get_kernel();

			REQUIRE(
			    kernel, "%s::set_trained_machine(): No kernel assigned to "
			            "machine of type \"%s\"\n",
			    get_name(), fold->get_trained_machine()->get_name());

			CCombinedKernel* combined_kernel =
			    dynamic_cast<CCombinedKernel*>(kernel);
			REQUIRE(
			    combined_kernel,
			    "%s::set_trained_machine(): This method is only"
			    " usable with CCombinedKernel on machines. This one is \"s\"\n",
			    get_name(), kernel->get_name());

			SGVector<float64_t> w = combined_kernel->get_subkernel_weights();

			/* evtl allocate memory (first call) */
			if (!m_mkl_weights.matrix)
			{
				SG_SDEBUG("allocating memory for mkl weight matrix\n")
				m_mkl_weights = SGMatrix<float64_t>(
				    w.vlen, obs->get_num_folds() * obs->get_num_runs());
			}

			/* put current mkl weights into matrix, copy memory vector wise to
			 * make
			 * things fast. Compute index of address to where vector goes */

			/* number of runs is w.vlen*m_num_folds shift */
			index_t run_shift =
			    fold->get_current_run_index() * w.vlen * obs->get_num_folds();

			/* fold shift is m_current_fold_index*w-vlen */
			index_t fold_shift = fold->get_current_fold_index() * w.vlen;

			/* add both index shifts */
			index_t first_idx = run_shift + fold_shift;
			SG_SDEBUG(
			    "run %d, fold %d, matrix index %d\n",
			    fold->get_current_run_index(), fold->get_current_fold_index(),
			    first_idx);

			/* copy memory */
			sg_memcpy(
			    &m_mkl_weights.matrix[first_idx], w.vector,
			    w.vlen * sizeof(float64_t));

			SG_UNREF(kernel);
		}
	}
}
