/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef _PNORM_ONE__H__
#define _PNORM_ONE__H__

#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>

#include <stdio.h>

namespace shogun
{
/** @brief Preprocessor PNorm, normalizes vectors to have p-norm.
 *
 * Formally, it computes
 *
 * \f[
 * {\bf x} \leftarrow \frac{{\bf x}}{||{\bf x}||_p}
 * \f]
 *
 */
class CPNorm : public CSimplePreprocessor<float64_t>
{
	public:
		/** default PNorm Constructor */
		CPNorm ();

		CPNorm (double p);
		
		/** destructor */
		virtual ~CPNorm ();

		/// initialize preprocessor from features
		virtual bool init (CFeatures* features);
		/// cleanup
		virtual void cleanup ();
		/// initialize preprocessor from file
		virtual bool load (FILE* f);
		/// save preprocessor init-data to file
		virtual bool save (FILE* f);

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGMatrix<float64_t> apply_to_feature_matrix (CFeatures* features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector (SGVector<float64_t> vector);

		/** @return object name */
		virtual inline const char* get_name () const { return "PNorm"; }

		/// return a type of preprocessor
		virtual inline EPreprocessorType get_type () const { return P_PNORM; }
		
		/**
		 * Set norm
		 * @param p norm value
		 */
		void setPNorm (double pnorm);
		
		/**
		 * Get norm
		 * @return norm
		 */
		double getPNorm () const;
	
	private: 
		void register_param ();
		inline float64_t getPNorm (float64_t* vec, int32_t vec_len) const;
		
	private:
		double m_p;
};
}
#endif /* _PNORM_ONE__H__ */
