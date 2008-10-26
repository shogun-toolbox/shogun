/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LOCALALIGNMENTSTRINGKERNEL_H___
#define _LOCALALIGNMENTSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

#define LOGSUM_TBL 10000      /* span of the logsum table */

/** The LocalAlignmentString kernel compares two sequences through all possible
 * local alignments between the two sequences. The implementation is taken from
 * http://www.mloss.org/software/view/40/ and only adjusted to work with shogun.
 */
class CLocalAlignmentStringKernel: public CStringKernel<char>
{
	public:
		/** constructor
		 * @param size cache size
		 */
		CLocalAlignmentStringKernel(int32_t size);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CLocalAlignmentStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r);

		virtual ~CLocalAlignmentStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		virtual bool load_init(FILE* src) { return false; }

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		virtual bool save_init(FILE* dest) { return false; }

		/** return what type of kernel we are
		 *
		 * @return kernel type LOCALALIGNMENT
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_LOCALALIGNMENT;
		}

		/** return the kernel's name
		 *
		 * @return name LocalAlignment
		 */
		virtual const char* get_name()
		{
			return "LocalAlignment";
		}

	private:
		/** initialize logarithmic sum */
		void init_logsum();

		/** logarithmic sum
		 *
		 * @param p1 parameter1
		 * @param p2 parameter2
		 * @return logarithmic sum as integer
		 */
		int32_t LogSum(int32_t p1, int32_t p2);

		/** logarithmic sum 2
		 *
		 * @param p1 parameter 1
		 * @param p2 parameter 2
		 * @return logarithmic sum as floating point
		 */
		float LogSum2(float p1, float p2);

		/** initialize */
		void initialize();

		/** LAkernel compute
		 *
		 * @param aaX aaX
		 * @param aaY aaY
		 * @param nX nX
		 * @param nY nY
		 * @return computed value
		 */
		DREAL LAkernelcompute(
			int32_t* aaX, int32_t* aaY, int32_t nX, int32_t nY);

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual DREAL compute(int32_t idx_a, int32_t idx_b);

	protected:
		/** if kernel is initialized */
		bool initialized;

		/** indicates whether a char is an amino-acid */
		int32_t *isAA;
		/** correspondance between amino-acid letter and index */
		int32_t *aaIndex;

		/** gap penalty opening */
		int32_t opening;
		/** gap penalty extension */
		int32_t extension;

		/** static logsum lookup */
		static int32_t logsum_lookup[LOGSUM_TBL];
		/** static blosum */
		static const int32_t blosum[];
		/** scaled blosum */
		int32_t* scaled_blosum;
};

#endif /* _LOCALALIGNMENTSTRINGKERNEL_H__ */

