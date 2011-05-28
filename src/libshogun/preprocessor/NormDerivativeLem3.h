/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CNORM_DERIVATIVE_LEM3__H__
#define _CNORM_DERIVATIVE_LEM3__H__

#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>
namespace shogun
{
/** @brief Preprocessor NormDerivativeLem3, performs the normalization used in Lemma3
 * in Jaakola Hausslers Fischer Kernel paper <b>currently not implemented</b>*/
class CNormDerivativeLem3 : public CSimplePreprocessor<float64_t>
{
	public:
		/** default constructor */
		CNormDerivativeLem3();

		/** destructor */
		virtual ~CNormDerivativeLem3();

		/// initialize preprocessor from features
		virtual bool init(CFeatures* f);
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
		virtual float64_t* apply_to_feature_vector(float64_t* f, int32_t& len);

		/** @return object name */
		virtual inline const char* get_name() const { return "NormDerivativeLem3"; }

		/// return a type of preprocessor
		virtual inline EPreprocessorType get_type() const { return P_NORMDERIVATIVELEM3; }
};
}
#endif
