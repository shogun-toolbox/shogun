#ifndef _SHORTKERNEL_H___
#define _SHORTKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CShortKernel : public CSimpleKernel<SHORT>
{
	public:
		CShortKernel(long cachesize) : CSimpleKernel<SHORT>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<SHORT>::init(l,r, do_init);

			assert(l->get_feature_type()==F_SHORT);
			assert(r->get_feature_type()==F_SHORT);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_SHORT; }
};
#endif
