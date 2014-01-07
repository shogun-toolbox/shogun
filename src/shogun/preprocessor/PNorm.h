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

#include <preprocessor/DensePreprocessor.h>
#include <features/Features.h>
#include <lib/common.h>

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
class CPNorm : public CDensePreprocessor<float64_t>
{
	public:
		/** default PNorm Constructor */
		CPNorm ();

    /** constructor
     * @param p the norm to calculate. NOTE: has to be greater or equal than 1.0
     */
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
		virtual const char* get_name () const { return "PNorm"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type () const { return P_PNORM; }

		/**
		 * Set norm
		 * @param pnorm norm value
		 */
		void set_pnorm (double pnorm);

		/**
		 * Get norm value
		 * @return norm
		 */
		double get_pnorm () const;

	private:
		void register_param ();
		inline float64_t get_pnorm (float64_t* vec, int32_t vec_len) const;

	private:
		double m_p;
};
}
#endif /* _PNORM_ONE__H__ */
