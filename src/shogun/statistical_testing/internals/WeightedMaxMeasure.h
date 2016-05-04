/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Heiko Strathmann
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

#ifndef WEIGHTED_MAX_MEASURE_H__
#define WEIGHTED_MAX_MEASURE_H__

#include <shogun/lib/common.h>
#include <shogun/statistical_testing/internals/MaxMeasure.h>

namespace shogun
{

class CKernel;
class CMMD;

namespace internal
{

class WeightedMaxMeasure : public MaxMeasure
{
public:
	WeightedMaxMeasure(KernelManager&, CMMD*);
	WeightedMaxMeasure(const WeightedMaxMeasure& other)=delete;
	~WeightedMaxMeasure();
	WeightedMaxMeasure& operator=(const WeightedMaxMeasure& other)=delete;
	virtual CKernel* select_kernel() override;
};

}

}

#endif // WEIGHTED_MAX_MEASURE_H__
