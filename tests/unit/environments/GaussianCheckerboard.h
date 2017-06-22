/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 MikeLing, Viktor Gal, Sergey Lisitsyn, Heiko Strathmann
 */

#ifndef GAUSSIANCHECKERBOARD_HPP
#define GAUSSIANCHECKERBOARD_HPP

#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

class GaussianCheckerboard
{
public:
	GaussianCheckerboard(
	    const int32_t num_samples, const int32_t num_labels,
	    const int32_t num_dim)
	{
		ASSERT(num_labels > 1)
		SGMatrix<float64_t> data = CDataGenerator::generate_gaussians(
		    num_samples, num_labels, num_dim);
		CDenseFeatures<float64_t> features(data);

		set_size = data.num_cols / 2;
		SGVector<index_t> train_idx(set_size), test_idx(set_size);
		SGVector<float64_t> labels(set_size);
		for (index_t i = 0, j = 0; i < data.num_cols; ++i)
		{
			if (i % 2 == 0)
				train_idx[j] = i;
			else
				test_idx[j++] = i;
		}

		// it's going to generate binary label data
		if (num_labels == 2)
		{
			for (index_t i = 0; i < data.num_cols; ++i)
			{
				labels[i / 2] = (i < data.num_cols / 2) ? 1.0 : -1.0;
			}

			labels_train = new CBinaryLabels(labels);
			labels_test = new CBinaryLabels(labels);
		}
		if (num_labels > 2)
		{
			int32_t step = (data.num_cols) / (2 * num_labels);
			for (int32_t l = 0; l < num_labels; l++)
			{
				for (index_t i = l * step;
				     (i < (l + 1) * step) && (i < data.num_cols / 2); ++i)
				{
					labels[i] = l;
				}
			}

			labels_train = new CMulticlassLabels(labels);
			labels_test = new CMulticlassLabels(labels);
		}

		features_train =
		    (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
		features_test =
		    (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

		SG_REF(features_train)
		SG_REF(features_test)
		SG_REF(labels_train)
		SG_REF(labels_test)
	}

	~GaussianCheckerboard()
	{
		SG_UNREF(features_train)
		SG_UNREF(features_test)
		SG_UNREF(labels_train)
		SG_UNREF(labels_test)
	}

	/* get the traning features */
	CDenseFeatures<float64_t>* get_features_train() const
	{
		return features_train;
	}

	/* get the test features */
	CDenseFeatures<float64_t>* get_features_test() const
	{
		return features_test;
	}

	/* get the test labels */
	CLabels* get_labels_train() const
	{
		return labels_train;
	}

	/* get the traning labels */
	CLabels* get_labels_test() const
	{
		return labels_test;
	}

	/* return the size of data set */
	int32_t get_set_size() const
	{
		return set_size;
	}

protected:
	// data for training
	CDenseFeatures<float64_t>* features_train;

	// data for testing
	CDenseFeatures<float64_t>* features_test;

	// traning label
	CLabels* labels_train;

	// testing label
	CLabels* labels_test;

	// the size of generated data set
	int32_t set_size;
};
#endif
