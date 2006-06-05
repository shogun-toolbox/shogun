#ifndef _DREALKERNEL_H___
#define _DREALKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CRealKernel : public CSimpleKernel<DREAL>
{
	public:
		CRealKernel(LONG cachesize) : CSimpleKernel<DREAL>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<DREAL>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_DREAL);
			ASSERT(r->get_feature_type()==F_DREAL);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }
};
#endif
