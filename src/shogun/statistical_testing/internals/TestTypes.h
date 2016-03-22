/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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

#ifndef TEST_TYPES_H__
#define TEST_TYPES_H__

namespace shogun
{

namespace internal
{

/**
 * @brief Meta test-type for 1-distribution statistical tests.
 */
struct OneDistributionTest
{
	/** defines the number of feature objects required */
	static constexpr index_t num_feats = 1;
};

/**
 * @brief Meta test-type for 2-distribution statistical tests.
 */
struct TwoDistributionTest
{
	/** defines the number of feature objects required */
	static constexpr index_t num_feats = 2;
};

/**
 * @brief Meta test-type for 3-distribution statistical tests.
 */
struct ThreeDistributionTest
{
	/** defines the number of feature objects required */
	static constexpr index_t num_feats = 3;
};

/**
 * @brief Meta test-type for goodness-of-fit test.
 */
struct GoodnessOfFitTest : OneDistributionTest
{
	/** defines the number of kernel objects required */
	static constexpr index_t num_kernels = 1;
};

/**
 * @brief Meta test-type for two-sample test.
 */
struct TwoSampleTest : TwoDistributionTest
{
	/** defines the number of kernel objects required */
	static constexpr index_t num_kernels = 1;
};

/**
 * @brief Meta test-type for independence test.
 */
struct IndependenceTest : TwoDistributionTest
{
	/** defines the number of kernel objects required */
	static constexpr index_t num_kernels = 2;
};

}

}

#endif // TEST_TYPES_H__
