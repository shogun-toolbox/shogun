/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CNORM_ONE__H__
#define _CNORM_ONE__H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>

#include <stdio.h>

namespace shogun
{
/** @brief Preprocessor NormOne, normalizes vectors to have norm 1.
 *
 * Formally, it computes
 *
 * \f[
 * {\bf x} \leftarrow \frac{{\bf x}}{||{\bf x}||}
 * \f]
 *
 * It therefore does not need any initialization. It is most useful to get data
 * onto a ball of radius one.
 */
class CNormOne : public CDensePreprocessor<float64_t>
{
	public:
		/** default constructor */
		CNormOne();

		/** destructor */
		virtual ~CNormOne();

		/// initialize preprocessor from features
		virtual bool init(CFeatures* features);
		/// cleanup
		virtual void cleanup();
		/// initialize preprocessor from file
		virtual bool load(FILE* f);
		/// save preprocessor init-data to file
		virtual bool save(FILE* f);

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name() const { return "NormOne"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const { return P_NORMONE; }
};
}
#endif
