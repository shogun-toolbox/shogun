/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

namespace shogun
{

	class CFeatures;

	/** @brief class Transformer used to transform data
	 *
	 */
	class CTransformer : public CSGObject
	{
	public:
		/** constructor */
		CTransformer() : CSGObject()
		{
		}

		/** destructor */
		virtual ~CTransformer()
		{
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "Transformer";
		}

		/** Fit transformer to features */
		virtual void fit(CFeatures* features)
		{
		}

		/** Fit transformer to features and labels */
		virtual void fit(CFeatures* features, CLabels* labels)
		{
			SG_SNOTIMPLEMENTED;
		}

		/** Apply transform to features. If transform is performed in place,
		 *  underlying data of input features will be reused if possible.
		 *	@param features features to transform
		 *	@param inplace whether transform in place
		 *	@return the result feature object after applying the transformer
		 * */
		virtual CFeatures* apply(CFeatures* features, bool inplace = true) = 0;
	};
}
#endif /* TRANSFORMER_H_ */
