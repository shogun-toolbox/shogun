/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __STREAMINGKERNEL_H_
#define __STREAMINGKERNEL_H_

#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CStreamingFeatures;

class CStreamingKernel: public CKernel
{
	public:
		CStreamingKernel();

		CStreamingKernel(CStreamingFeatures* streaming_lhs,
				CStreamingFeatures* streaming_rhs, CKernel* baseline_kernel);

		virtual ~CStreamingKernel();

		/** @return type of kernel */
		inline virtual EKernelType get_kernel_type()
		{
			return K_POWER;
		}

		/** @return type of features */
		inline virtual EFeatureType get_feature_type()
		{
			return lhs->get_feature_type();
		}

		/** @return class of features */
		inline virtual EFeatureClass get_feature_class()
		{
			return lhs->get_feature_class();
		}

		virtual SGMatrix<float64_t> get_kernel_matrix();

		void set_blocksize(index_t blocksize) { m_block_size=blocksize; }

		/** Overloaded init function, makes sure provided CFeatures are
		 * CStreamingFeatures and calls superclass method
		 *
		 * @param lhs features for left-hand side
		 * @param rhs features for right-hand side
		 * @return if init was successful
		 */
		bool init(CFeatures* l, CFeatures* r);

		/** @return name of SG_SERIALIZABLE */
		virtual const char* get_name() const { return "StreamingKernel"; }

	protected:
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		CKernel* m_baseline_kernel;

		CStreamingFeatures* m_streaming_lhs;
		CStreamingFeatures* m_streaming_rhs;

		index_t m_block_size;
};
}
#endif /* __STREAMINGKERNEL_H_ */
