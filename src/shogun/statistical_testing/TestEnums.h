/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#ifndef TEST_ENUMS_H_
#define TEST_ENUMS_H_

#include <shogun/lib/config.h>

namespace shogun
{

enum EStatisticType
{
	ST_UNBIASED_FULL,
	ST_UNBIASED_INCOMPLETE,
	ST_BIASED_FULL
};

enum EVarianceEstimationMethod
{
	VEM_DIRECT,
	VEM_PERMUTATION
};

enum ENullApproximationMethod
{
	NAM_PERMUTATION,
	NAM_MMD1_GAUSSIAN,
	NAM_MMD2_SPECTRUM,
	NAM_MMD2_GAMMA
};

enum EKernelSelectionMethod
{
	KSM_MEDIAN_HEURISTIC,
	KSM_MAXIMIZE_MMD,
	KSM_MAXIMIZE_POWER,
	KSM_CROSS_VALIDATION,
	KSM_AUTO = KSM_MAXIMIZE_POWER
};

}
#endif // TEST_ENUMS_H_
