#ifndef _REALKERNEL_H___
#define _REALKERNEL_H___

#include "kernel/SimpleKernel.h"

class CRealKernel : public CSimpleKernel<REAL>
{
	public:
		CRealKernel(long cachesize) : CSimpleKernel<REAL>(cachesize)
		{
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_REAL; }
};
#endif
