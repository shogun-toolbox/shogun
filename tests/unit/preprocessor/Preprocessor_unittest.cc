/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/SortWordString.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(Preprocessor, dense_apply)
{
	const index_t dim=2;
	const index_t size=4;
	SGMatrix<float64_t> data(dim, size);
	for (index_t i=0; i<dim; ++i)
	{
		for (index_t j=0; j<size; ++j)
			data(i, j)=CMath::randn_double();
	}

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CDensePreprocessor<float64_t>* preproc=new CNormOne();
	preproc->init(features);

	CFeatures* preprocessed=preproc->apply(features);

	ASSERT(preprocessed);
	EXPECT_EQ(preprocessed->get_feature_class(), C_DENSE);

	SG_UNREF(preproc);
	SG_UNREF(preprocessed);
	SG_UNREF(features);
}

TEST(Preprocessor, string_apply)
{
	const index_t num_strings=3;
	const index_t max_string_length=20;
	const index_t min_string_length=max_string_length/2;

	SGStringList<uint16_t> strings(num_strings, max_string_length);

	for (index_t i=0; i<num_strings; ++i)
	{
		index_t len=CMath::random(min_string_length, max_string_length);
		SGString<uint16_t> current(len);

		/* fill with random uppercase letters (ASCII) */
		for (index_t j=0; j<len; ++j)
		{
			current.string[j]=(uint16_t)CMath::random('A', 'Z');

			/* attach \0 to print letter */
			uint16_t* string=SG_MALLOC(uint16_t, 2);
			string[0]=current.string[j];
			string[1]='\0';
			SG_FREE(string);
		}

		strings.strings[i]=current;
	}

	/* create num_features 2-dimensional vectors */
	CStringFeatures<uint16_t>* features=new CStringFeatures<uint16_t>(strings, ALPHANUM);
	CStringPreprocessor<uint16_t>* preproc=new CSortWordString();
	preproc->init(features);

	CFeatures* preprocessed=preproc->apply(features);

	ASSERT(preprocessed);
	EXPECT_EQ(preprocessed->get_feature_class(), C_STRING);

	SG_UNREF(preproc);
	SG_UNREF(preprocessed);
	SG_UNREF(features);
}
