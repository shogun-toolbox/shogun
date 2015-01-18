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

#include <shogun/statistics/KernelSelection.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/statistics/KernelTwoSampleTest.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/statistics/QuadraticTimeMMD.h>

using namespace shogun;

CKernelSelection::CKernelSelection()
{
	init();
}

CKernelSelection::CKernelSelection(CKernelTwoSampleTest* estimator)
{
	init();
	set_estimator(estimator);
}

CKernelSelection::~CKernelSelection()
{
	SG_UNREF(m_estimator);
}

void CKernelSelection::init()
{
	SG_ADD((CSGObject**)&m_estimator, "estimator",
			"Underlying CKernelTwoSampleTest instance", MS_NOT_AVAILABLE);

	m_estimator=NULL;
}

void CKernelSelection::set_estimator(CKernelTwoSampleTest* estimator)
{
	REQUIRE(estimator, "No CKernelTwoSampleTest instance provided!\n");

	/* ensure that there is a combined kernel */
	CKernel* kernel=estimator->get_kernel();
	REQUIRE(kernel, "Underlying \"%s\" has no kernel set!\n",
			estimator->get_name());
	REQUIRE(kernel->get_kernel_type()==K_COMBINED, "Kernel of underlying \"%s\" "
			"is of type \"%s\" but is has to be CCombinedKernel\n",
			estimator->get_name(), kernel->get_name());
	SG_UNREF(kernel);

	SG_REF(estimator);
	SG_UNREF(m_estimator);
	m_estimator=estimator;
}

CKernelTwoSampleTest* CKernelSelection::get_estimator() const
{
	SG_REF(m_estimator);
	return m_estimator;
}
