/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 20013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#ifndef __RESCALEFEATURES_H__
#define __RESCALEFEATURES_H__

#include <preprocessor/DensePreprocessor.h>

namespace shogun
{
	/**@brief Preprocessor RescaleFeautres is rescaling the range of features to
	 * make the features independent of each other and aims to scale the range
	 * in [0, 1] or [-1, 1].
	 *
	 * The general formula is given as:
	 * \f[
	 * x' = \frac{x - min}{max - min}
	 * \f]
	 * where \f$x\f$ is an original value, \f$x'\f$ is the normalized value.
     */
	class CRescaleFeatures : public CDensePreprocessor<float64_t>
	{
		public:
			/** default ctor */
			CRescaleFeatures();

			/** dtor */
			virtual ~CRescaleFeatures();

			/**
			 * initialize preprocessor from features
			 *
			 * @param features the features to derive the min and max values from.
			 */
			virtual bool init(CFeatures* features);

			/**
			 * Cleanup
			 */
			virtual void cleanup();

			/**
			 * Apply preproc on a feature matrix
			 *
			 * @param features input feature matrix
			 * @return pointer to feature_matrix, i.e. f->get_feature_matrix();
			 */
			virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

			/**
			 * Apply preproc on a single feature vector
			 */
			virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

			/** @return object name */
			virtual const char* get_name() const { return "RescaleFeatures"; }

			/** return a type of preprocessor */
			virtual EPreprocessorType get_type() const { return P_RESCALEFEATURES; }

		private:
			void register_parameters();

		protected:
			/** min */
			SGVector<float64_t> m_min;
			/** 1.0/(max[i]-min[i]) */
			SGVector<float64_t> m_range;
			/** true when already initialized */
			bool m_initialized;
		};
}

#endif /* __RESCALEFEATURES_H__ */
