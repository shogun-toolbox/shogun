/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013-2014 Soumyajit De
 * Copyright (C) 2013-2014 Indian Institute of Technology, Bombay
 */

#ifndef _SSKSTRINGKERNEL_H___
#define _SSKSTRINGKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief
 */
class CSSKStringKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CSSKStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param maxlen maximum length of the subsequence
		 * @param lambda the penalty parameter
		 */
		CSSKStringKernel(int32_t size, int32_t maxlen, float64_t lambda);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param maxlen maximum length of the subsequence
		 * @param lambda the penalty parameter
		 */
		CSSKStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r,
			int32_t maxlen, float64_t lambda);

		virtual ~CSSKStringKernel();

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
		 * @return kernel type POLYMATCH
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_POLYMATCH;
		}

		/** return the kernel's name
		 *
		 * @return name PolyMatchString
		 */
		virtual const char* get_name() const { return "SSKStringKernel"; }

		/* register the parameters
		 */
		virtual void register_params();

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	protected:
		// maximum length of common subsequences
		int32_t m_maxlen;
		// gap penalty
		float64_t m_lambda;
		// default values of params
		static const int32_t DEFAULT_MAXLEN = 2;
  		static const float64_t DEFAULT_LAMBDA = 0.75;

	private:
		void init();
};
}
#endif /* _SSKSTRINGKERNEL_H___ */
