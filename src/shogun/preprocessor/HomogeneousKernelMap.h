/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal, Evgeniy Andreev, Yuyu Zhang, 
 *          Evan Shelhamer, Soeren Sonnenburg, Saurabh Goyal
 */

#ifndef _CHOMOGENEOUKERNELMAP__H__
#define _CHOMOGENEOUKERNELMAP__H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>


namespace shogun
{
	/** @brief Type of kernel */
	enum HomogeneousKernelType {
	  HomogeneousKernelIntersection = 0, /**< intersection kernel */
	  HomogeneousKernelChi2, /**< Chi2 kernel */
	  HomogeneousKernelJS /**< Jensen-Shannon kernel */
	};

	/** @brief Type of spectral windowing function  */
	enum HomogeneousKernelMapWindowType {
	  HomogeneousKernelMapWindowUniform = 0, /**< uniform window */
	  HomogeneousKernelMapWindowRectangular = 1, /**< rectangular window */
	};

	/** @brief Preprocessor HomogeneousKernelMap performs homogeneous kernel maps
	 * as described in
	 *
	 * A. Vedaldi and A. Zisserman.
	 * Efficient additive kernels via explicit feature maps.
	 * In PAMI, 2011
	 *
	 * The homogeneous kernel map is a finite dimensional linear
	 * approximation of homogeneous kernels, including the intersection,
	 * chi-squared, and Jensen-Shannon kernels. These kernels
	 * are frequently used in computer vision applications because they
	 * are particular suitable for data in the format of histograms, which
	 * encompasses many visual descriptors used.
	 *
	 * The implementation is unsafe to work with negative feature values.
	 *
	 * Implementation is based on the vlfeat library.
	 *
	 */
	class HomogeneousKernelMap : public DensePreprocessor<float64_t>
	{
		public:
			/** default constructor */
			HomogeneousKernelMap();

			/** constructor
			 *
			 * @param kernel kernel type
			 * @param wType window type (use HomogeneousKernelMapWindowRectangular if unsure)
			 * @param gamma the homogeneity order
			 * @param order the approximation order
			 * @param period the period (use a negative value to use the default period)
			 */
			HomogeneousKernelMap(HomogeneousKernelType kernel, HomogeneousKernelMapWindowType wType,
			                      float64_t gamma = 1.0, uint64_t order = 1, float64_t period = -1);

			/** destructor */
			virtual ~HomogeneousKernelMap();

			/** cleanup */
			virtual void cleanup();

			/** applies to feature vector
			 * @param vector features vector
			 * @return transformed feature vector
			 */
			virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

			/** @return object name */
			virtual const char* get_name() const { return "HomogeneousKernelMap"; }

			/** @return a type of preprocessor */
			virtual EPreprocessorType get_type() const { return P_HOMOGENEOUSKERNELMAP; }

			/** sets kernel type
			 * @param k type of homogeneous kernel
			 */
			void set_kernel_type(HomogeneousKernelType k);
			/** returns kernel type
			 * @return kernel type
			 */
			HomogeneousKernelType get_kernel_type() const;

			/** sets window type
			 * @param w type of window
			 */
			void set_window_type(HomogeneousKernelMapWindowType w);
			/** returns window type
			 * @return window type
			 */
			HomogeneousKernelMapWindowType get_window_type() const;

			/** sets gamma
			 * @param g gamma value
			 */
			void set_gamma(float64_t g);
			/** returns gamma
			 * @return gamma value
			 */
			float64_t get_gamma(float64_t g) const;

			/** sets approximation order
			 * @param o order
			 */
			void set_order(uint64_t o);
			/** returns approximation order
			 * @return approximation order
			 */
			uint64_t get_order() const;

			/** sets period
			 * @param p period value
			 */
			void set_period(float64_t p);
			/** returns period value
			 * @return period value
			 */
			float64_t get_period() const;

		protected:
		    virtual SGMatrix<float64_t>
		    apply_to_matrix(SGMatrix<float64_t> matrix);

		private:
			void init ();
			void register_params ();
			inline float64_t get_smooth_spectrum (float64_t omega) const;
			inline float64_t sinc (float64_t x) const;
			inline float64_t get_spectrum (float64_t omega) const;
			SGVector<float64_t> apply_to_vector(const SGVector<float64_t>& in_v) const;

		private:
			HomogeneousKernelType m_kernel;
			HomogeneousKernelMapWindowType m_window;
			float64_t m_gamma;
			float64_t m_period;
			uint64_t m_numSubdivisions;
			float64_t m_subdivision;
			uint64_t m_order;
			int64_t m_minExponent;
			int64_t m_maxExponent;
			SGVector<float64_t> m_table;
	};
}
#endif /* _CHOMOGENEOUKERNELMAP__H__ */
