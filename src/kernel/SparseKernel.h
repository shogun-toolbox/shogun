#ifndef _SPARSEKERNEL_H___
#define _SPARSEKERNEL_H___

#include "kernel/Kernel.h"
#include "features/SparseFeatures.h"

template <class ST> class CSparseKernel : public CKernel
{
	public:
		CSparseKernel(LONG cachesize) : CKernel(cachesize)
		{
		}

		/** initialize your kernel
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 * when training data is supplied as both l and r do_init should
		 * be true; when testing it must be false and thus no further
		 * initialization of the preprocessor in the kernl
		 * will be done (like determining the scale factor when rescaling the kernel).
		 * instead the previous values will be used (which where hopefully obtained
		 * on training data/loaded)
		 */
		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CKernel::init(l,r,do_init);

			assert(l->get_feature_class() == C_SPARSE);
			assert(r->get_feature_class() == C_SPARSE);
			lhs=(CSparseFeatures<ST>*) l;
			rhs=(CSparseFeatures<ST>*) r;

			return true;
		}

		/** return feature class the kernel can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_SPARSE; }
};
#endif
