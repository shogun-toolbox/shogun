#ifndef _STRINGKERNEL_H___
#define _STRINGKERNEL_H___

#include "kernel/Kernel.h"
#include "features/StringFeatures.h"

template <class ST> class CStringKernel : public CKernel
{
	public:
		CStringKernel(LONG cachesize) : CKernel(cachesize)
		{
		}

		/** initialize your kernel
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 * when training data is supplied as both l and r do_init should
		 * be true; when testing it must be false and thus no further
		 * initialization of the preprocessor in the kernel
		 * will be done (like determining the scale factor when rescaling the kernel).
		 * instead the previous values will be used (which where hopefully obtained
		 * on training data/loaded)
		 */
		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CKernel::init(l,r,do_init);

			assert(l->get_feature_class() == C_STRING);
			assert(r->get_feature_class() == C_STRING);

			return true;
		}

		/** return feature class the kernel can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }
};
#endif

