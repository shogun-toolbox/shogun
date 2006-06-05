#ifndef _WORDKERNEL_H___
#define _WORDKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CWordKernel : public CSimpleKernel<WORD>
{
	public:
		CWordKernel(LONG cachesize) : CSimpleKernel<WORD>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<WORD>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_WORD);
			ASSERT(r->get_feature_type()==F_WORD);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_WORD; }
};
#endif
