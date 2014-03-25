/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Sebastian J. Schultheiss and Soeren Sonnenburg
 * Copyright (C) 2009 Max-Planck-Society
 */

#ifndef _REGULATORYMODULESSTRINGKERNEL_H___
#define _REGULATORYMODULESSTRINGKERNEL_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The Regulaty Modules kernel, based on the WD kernel,
 * as published in Schultheiss et al., Bioinformatics (2009)
 * on regulatory sequences.
 *
 */
class CRegulatoryModulesStringKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CRegulatoryModulesStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width of gaussian kernel
		 * @param degree degree of wds kernel
		 * @param shift shift of wds kernel
		 * @param window size of window around motifs to compute wds kernels on
		 */
		CRegulatoryModulesStringKernel(int32_t size, float64_t width, int32_t degree, int32_t shift, int32_t window);

		/** constructor
		 *
		 * @param lstr string features of left-hand side
		 * @param rstr string features of right-hand side
		 * @param lpos motif positions on lhs
		 * @param rpos motif positions on rhs
		 * @param width width of gaussian kernel
		 * @param degree degree of wds kernel
		 * @param shift shift of wds kernel
		 * @param window size of window around motifs to compute wds kernels on
		 * @param size cache size
		 */
		CRegulatoryModulesStringKernel(CStringFeatures<char>* lstr, CStringFeatures<char>* rstr,
			CDenseFeatures<uint16_t>* lpos, CDenseFeatures<uint16_t>* rpos,
			float64_t width, int32_t degree, int32_t shift, int32_t window, int32_t size=10);

		/** default destructor */
		virtual ~CRegulatoryModulesStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type() { return K_REGULATORYMODULES; }

		/** return the kernel's name
		 *
		 * @return name Regulatory Modules
		 */
		virtual const char* get_name() const { return "RegulatoryModulesStringKernel"; }

		/** set motif positions
		 *
		 * @param positions_lhs motif positions on lhs
		 * @param positions_rhs motif positions on rhs
		 */
		void set_motif_positions(
			CDenseFeatures<uint16_t>* positions_lhs, CDenseFeatures<uint16_t>* positions_rhs);

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

		/** compute WDS kernel for features a and b
		 *
		 * @param avec vector a
		 * @param bvec vector b
		 * @param len length of string
		 * @return computed kernel function
		 */
		float64_t compute_wds(char* avec, char* bvec, int32_t len);


		/** set standard weighted degree kernel weighting */
		void set_wd_weights();

	private:
		/** initialises parameters and registers them */
		void init();

	protected:
		/** width of Gaussian kernel part */
		float64_t width;

		/** degree of Weighted Degree kernel part */
		int32_t degree;
		/** shift of Weighted Degree with Shifts kernel part */
		int32_t shift;

		/** size of window around motifs */
		int32_t window;

		/** Matrix of motif positions from sequences left-hand side */
		CDenseFeatures<uint16_t>* motif_positions_lhs;

		/** Matrix of motif positions from sequences right-hand side */
		CDenseFeatures<uint16_t>* motif_positions_rhs;

		/** scaling weights in window */
		SGVector<float64_t> position_weights;

		/** weights of WD kernel */
		SGVector<float64_t> weights;
};
}
#endif /* _REGULATORYMODULESSTRINGKERNEL_H__ */
