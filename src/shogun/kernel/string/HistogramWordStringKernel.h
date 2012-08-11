/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HISTOGRAMWORDKERNEL_H___
#define _HISTOGRAMWORDKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/classifier/PluginEstimate.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
	class CPluginEstimate;
	template <class T> class CStringFeatures;
/** @brief The HistogramWordString computes the TOP kernel on inhomogeneous
 * Markov Chains. */
class CHistogramWordStringKernel: public CStringKernel<uint16_t>
{
	public:
		/** default constructor  */
		CHistogramWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param pie plugin estimate
		 */
		CHistogramWordStringKernel(int32_t size, CPluginEstimate* pie);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param pie plugin estimate
		 */
		CHistogramWordStringKernel(
			CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r,
			CPluginEstimate* pie);

		virtual ~CHistogramWordStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type HISTOGRAM
		 */
		virtual EKernelType get_kernel_type() { return K_HISTOGRAM; }

		/** return the kernel's name
		 *
		 * @return name Histogram
		 */
		virtual const char* get_name() const { return "HistogramWordStringKernel" ; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b);

		/** compute index
		 *
		 * @param position position
		 * @param symbol symbol
		 * @return index at given position in given symbol
		 */
		inline int32_t compute_index(int32_t position, uint16_t symbol)
		{
			return position*num_symbols+symbol+1;
		}

	private:
		void init();

	protected:
		/** plugin estimate */
		CPluginEstimate* estimate;

		/** mean */
		float64_t* mean;
		/** variance */
		float64_t* variance;

		/** sqrt diagonal of left-hand side */
		float64_t* sqrtdiag_lhs;
		/** sqrt diagonal of right-hand side */
		float64_t* sqrtdiag_rhs;

		/** ld mean left-hand side */
		float64_t* ld_mean_lhs;
		/** ld mean right-hand side */
		float64_t* ld_mean_rhs;

		/** plo left-hand side */
		float64_t* plo_lhs;
		/** plo right-hand side */
		float64_t* plo_rhs;

		/** number of parameters */
		int32_t num_params;
		/** number of parameters2 */
		int32_t num_params2;
		/** number of symbols */
		int32_t num_symbols;
		/** sum m2 s2 */
		float64_t sum_m2_s2;

		/** if kernel is initialized */
		bool initialized;
};
}
#endif /* _HISTOGRAMWORDKERNEL_H__ */
