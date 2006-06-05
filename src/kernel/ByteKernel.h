#ifndef _BYTEKERNEL_H___
#define _BYTEKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CByteKernel : public CSimpleKernel<BYTE>
{
	public:
		CByteKernel(LONG cachesize) : CSimpleKernel<BYTE>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<BYTE>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_BYTE);
			ASSERT(r->get_feature_type()==F_BYTE);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_BYTE; }
};
#endif
