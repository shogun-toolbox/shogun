/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CNORM_ONE__H__
#define _CNORM_ONE__H__

#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

/** Preprocessor NormOne, normalizes vectors to have norm 1, i.e.
 *
 * \f[
 * {\bf x} \leftarrow \frac{{\bf x}}{||{\bf x}||}
 * \f]
 *
 * It therefore does not need any initialization. It is most useful to get data
 * onto a ball of radius one.
 */
class CNormOne : public CSimplePreProc<float64_t>
{
	public:
		/** default constructor */
		CNormOne();
		virtual ~CNormOne();

		/// initialize preprocessor from features
		virtual bool init(CFeatures* f);
		/// initialize preprocessor from file
		virtual bool load_init_data(FILE* src);
		/// save init-data (like transforamtion matrices etc) to file
		virtual bool save_init_data(FILE* dst);
		/// cleanup
		virtual void cleanup();
		/// initialize preprocessor from file
		virtual bool load(FILE* f);
		/// save preprocessor init-data to file
		virtual bool save(FILE* f);

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual float64_t* apply_to_feature_matrix(CFeatures* f);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual float64_t* apply_to_feature_vector(float64_t* f, int32_t &len);

		/** @return object name */
		inline virtual const char* get_name() { return "NormOne"; }
};
#endif
