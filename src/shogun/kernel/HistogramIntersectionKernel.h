/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Koen van de Sande
 * Copyright (C) 2010 Koen van de Sande / University of Amsterdam
 */

#ifndef _HISTOGRAMINTERSECTIONKERNEL_H___
#define _HISTOGRAMINTERSECTIONKERNEL_H___

#include <lib/common.h>
#include <kernel/DotKernel.h>
#include <features/DenseFeatures.h>

namespace shogun
{
/** @brief The HistogramIntersection kernel operating on realvalued vectors computes
 * the histogram intersection distance between sets of histograms.
 * Note: the current implementation assumes positive values for the histograms,
 * and input vectors should sum to 1.
 *
 * It is defined as
 * \f[
 * k({\bf x},{\bf x'})= \sum_{i=0}^{l} \mbox{min}(x^{\beta}_i, x'^{\beta}_i)
 * \f]
 * with \f$\beta=1\f$ by default
 * */
class CHistogramIntersectionKernel: public CDotKernel
{
	public:
		/** default constructor  */
		CHistogramIntersectionKernel();

		/** constructor
		 *
		 * @param size cache size
		 */
		CHistogramIntersectionKernel(int32_t size);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param beta kernel parameter
		 * @param size cache size
		 */
		CHistogramIntersectionKernel(
			CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
			float64_t beta=1.0, int32_t size=10);

		virtual ~CHistogramIntersectionKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);
		/* register the parameters
		 */
		virtual void register_params();

		/** return what type of kernel we are
		 *
		 * @return kernel type HISTOGRAMINTERSECTION
		 */
		virtual EKernelType get_kernel_type() { return K_HISTOGRAMINTERSECTION; }

		/** return the kernel's name
		 *
		 * @return name HistogramIntersectionKernel
		 */
		virtual const char* get_name() const { return "HistogramIntersectionKernel"; }

		/** getter for beta parameter
		 * @return beta value
		 */
		inline float64_t get_beta() { return m_beta; }

		/** setter for beta parameter
		 *  @param beta beta value
		 */
		inline void set_beta(float64_t beta) { m_beta = beta; }

	protected:

		/// beta parameter
		float64_t m_beta;

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

};
}
#endif /* _HISTOGRAMINTERSECTIONKERNEL_H__ */
