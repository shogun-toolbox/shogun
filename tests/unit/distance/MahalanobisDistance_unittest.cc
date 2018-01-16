/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2018 Wuwei Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the Shogun Development Team.
 */

#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/distance/MahalanobisDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/common.h>

using namespace shogun;

TEST(MahalanobisDistance, compute_distance)
{
	// create four points (0,0) (2,10) (2,8) (5,5)
	SGMatrix<float64_t> rect(2, 4);
	rect(0, 0) = 0;
	rect(1, 0) = 0;
	rect(0, 1) = 2;
	rect(1, 1) = 10;
	rect(0, 2) = 2;
	rect(1, 2) = 8;
	rect(0, 3) = 5;
	rect(1, 3) = 5;

	auto feature = some<CDenseFeatures<float64_t>>(rect);
	auto distance = some<CMahalanobisDistance>(feature, feature);
	EXPECT_NEAR(distance->distance(1, 1), 0.0, 1e-10);
	EXPECT_NEAR(distance->distance(1, 3), 2.63447126986, 1e-10);
	EXPECT_NEAR(distance->distance(2, 3), 2.22834405812, 1e-10);
}
