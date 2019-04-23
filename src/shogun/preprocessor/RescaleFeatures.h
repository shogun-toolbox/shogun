/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser
 */

#ifndef __RESCALEFEATURES_H__
#define __RESCALEFEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>

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
	class RescaleFeatures : public DensePreprocessor<float64_t>
	{
	public:
		/** default ctor */
		RescaleFeatures();

		/** dtor */
		virtual ~RescaleFeatures();

		/**
		 * Fit preprocessor into features
		 *
		 * @param features the features to derive the min and max values
		 * from.
		 */
		virtual void fit(std::shared_ptr<Features> features);

		/**
		 * Cleanup
		 */
		virtual void cleanup();

		/**
		 * Apply preproc on a single feature vector
		 */
		virtual SGVector<float64_t>
		apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name() const
		{
			return "RescaleFeatures";
		}

		/** return a type of preprocessor */
		virtual EPreprocessorType get_type() const
		{
			return P_RESCALEFEATURES;
		}

	private:
		void register_parameters();

	protected:
		virtual SGMatrix<float64_t> apply_to_matrix(SGMatrix<float64_t> matrix);

		/** min */
		SGVector<float64_t> m_min;
		/** 1.0/(max[i]-min[i]) */
		SGVector<float64_t> m_range;
	};
}

#endif /* __RESCALEFEATURES_H__ */
