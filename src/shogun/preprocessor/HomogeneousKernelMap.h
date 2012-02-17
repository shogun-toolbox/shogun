/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
 */

#ifndef _CHOMOGENEOUKERNELMAP__H__
#define _CHOMOGENEOUKERNELMAP__H__

#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>

#include <stdio.h>

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
	 *
	 * A. Vedaldi and A. Zisserman. 
	 * Efficient additive kernels via explicit feature maps. 
	 * In PAMI, 2011
	 *
	 * The homogeneous kernel map is a finite dimensional linear
	 * approximation of homgeneous kernels, including the intersection,
	 * chi-squared, and Jensen-Shannon kernels. These kernels
	 * are frequently used in computer vision applications because they
	 * are particular suitable for data in the format of histograms, which
	 * encompasses many visual descriptors used.
	 *
	 * Implementation is based on the vlfeat library.
	 *
	 */
	class CHomogeneousKernelMap : public CSimplePreprocessor<float64_t>
	{
		public:
			/** default constructor */
			CHomogeneousKernelMap ();

			/** constructor
			 *
			 * @param kernel kernel type
			 * @param window window type (use HomogeneousKernelMapWindowRectangular if unsure)
			 * @param gamma the homogeneity order
			 * @param order the approximation order
			 * @param period the period (use a negative value to use the default period)
			 */
			CHomogeneousKernelMap (HomogeneousKernelType kernel, HomogeneousKernelMapWindowType wType, 
														float64_t gamma = 1.0, uint64_t order = 1, float64_t period = -1);

			/** destructor */
			virtual ~CHomogeneousKernelMap ();
			
			/// initialize preprocessor from features
			virtual bool init(CFeatures* features);
			
			/// cleanup
			virtual void cleanup();
			
			virtual SGMatrix<float64_t> apply_to_feature_matrix (CFeatures* features);

			/// apply preproc on single feature vector
			/// result in feature matrix
			virtual SGVector<float64_t> apply_to_feature_vector (SGVector<float64_t> vector);

			/** @return object name */
			virtual inline const char* get_name () const { return "HomogeneousKernelMap"; }

			/// return a type of preprocessor
			virtual inline EPreprocessorType get_type () const { return P_HOMOGENEOUSKERNELMAP; }

			void setKernelType (HomogeneousKernelType k);
			HomogeneousKernelType getKernelType () const;
			
			void setWindowType (HomogeneousKernelMapWindowType w);
			HomogeneousKernelMapWindowType getWindowType () const;
			
			void setGamma (float64_t g);
			float64_t getGamma (float64_t g) const;
			
			void setOrder (uint64_t o);
			uint64_t getOrder () const;
			
			void setPeriod (float64_t p);
			float64_t getPeriod () const;
			
		private:
			void init ();
			void register_params ();
			inline float64_t get_smooth_spectrum (float64_t omega) const;
			inline float64_t sinc (float64_t x) const;
			inline float64_t get_spectrum (float64_t omega) const;
			inline void apply_to_vector (const SGVector<float64_t>& in_v, SGVector<float64_t>& out_v) const;
			
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