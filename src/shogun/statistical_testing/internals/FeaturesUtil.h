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

#ifndef FEATURES_UTIL_H__
#define FEATURES_UTIL_H__

#include <shogun/lib/common.h>

namespace shogun
{

class CFeatures;

namespace internal
{

/**
 * @brief Class FeaturesUtil for providing generic helper methods for
 * handling Shogun's feature objects for the big-testing framework.
 */
struct FeaturesUtil
{
	/**
	 * This creates a shallow copy of the feature object. It uses the same
	 * underlying feature storage as the original object, but it clones all
	 * the subsets.
	 *
	 * @param other The feature object whose shallow copy has to be created.
	 * @return A shallow copy of the feature object.
	 */
	static CFeatures* create_shallow_copy(CFeatures* other);

	/**
	 * This creates a merged copy of the two feature objects.
	 *
	 * @param feats_a First feature object.
	 * @param feats_b Second feature object.
	 * @return A merged copy of the feature objects with total number of feature
	 * vectors of feats_a.num_vectors+feats_b.num_vectors.
	 */
	static CFeatures* create_merged_copy(CFeatures* feats_a, CFeatures* feats_b);
};

}

}

#endif // FEATURES_UTIL_H__
