#ifndef _SIMPLEKERNEL_H___
#define _SIMPLEKERNEL_H___

#include "kernel/Kernel.h"
#include "features/SimpleFeatures.h"

template <class ST> class CSimpleKernel : public CKernel
{
	public:
		CSimpleKernel(long cachesize) : CKernel(cachesize)
		{
		}

		/** initialize your kernel
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 */
		virtual void init(CSimpleFeatures<ST>* l, CSimpleFeatures<ST>* r, bool do_init)
		{
			CKernel::init(l,r, do_init);
		}
};
#endif
