#ifndef _BYTEKERNEL_H___
#define _BYTEKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/io.h"

class CByteKernel : public CSimpleKernel<BYTE>
{
	public:
		CByteKernel(long cachesize) : CSimpleKernel<BYTE>(cachesize)
		{
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_BYTE; }
};
#endif
