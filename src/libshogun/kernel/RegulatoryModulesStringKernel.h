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

#include "lib/common.h"
#include "kernel/StringKernel.h"
#include "features/SimpleFeatures.h"

/** @brief The Regulaty Modules kernel, based on the WD kernel,
 * as published in Schultheiss et al., Bioinformatics (2009)
 * on regulatory sequences.
 *
 */
class CRegulatoryModulesStringKernel: public CStringKernel<char>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		CRegulatoryModulesStringKernel(int32_t size, float64_t width, int32_t degree, int32_t shift, int32_t window);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param size cache size
		 */
		CRegulatoryModulesStringKernel(CStringFeatures<char>* lstr, CStringFeatures<char>* rstr, 
			CSimpleFeatures<uint16_t>* lpos, CSimpleFeatures<uint16_t>* rpos, 
			float64_t width, int32_t degree, int32_t shift, int32_t window, int32_t size=10);

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
		 * @return kernel type GAUSSIAN
		 */
		virtual EKernelType get_kernel_type() { return K_REGULATORYMODULES; }

		/** return the kernel's name
		 *
		 * @return name Regulatory Modules
		 */
		inline virtual const char* get_name() const { return "RegulatoryModulesStringKernel"; }
		
		//FIXME
		void set_motif_positions(
			CSimpleFeatures<uint16_t>* positions_lhs, CSimpleFeatures<uint16_t>* positions_rhs);
		


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

		
		//FIXME
		void set_wd_weights();

	protected:
		/** width of Gaussian kernel part */
		float64_t width;

		/** degree of Weighted Degree kernel part */
		int32_t degree; 
		/** shift of Weighted Degree with Shifts kernel part */
		int32_t shift;
		
		//TODO
		int32_t window;

		/** Matrix of motif positions from sequences left-hand side */
		CSimpleFeatures<uint16_t>* motif_positions_lhs;

		/** Matrix of motif positions from sequences right-hand side */
		CSimpleFeatures<uint16_t>* motif_positions_rhs;

		float64_t* position_weights;

		float64_t* weights;

};

#endif /* _REGULATORYMODULESSTRINGKERNEL_H__ */
