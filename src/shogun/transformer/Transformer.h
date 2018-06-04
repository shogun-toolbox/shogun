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

	/** @brief class Transformer defines a transformer interface
	 *
	 * Transformers are transformation functions that transform input features
	 * to another features instance. As transformers might need a certain
	 * training on a training set (e.g. learning parameters), they may expect
	 * that the fit() function is called before anything else. The actual
	 * transformations happen in the transform() function, which returns the
	 * transformed features as a new CFeatures instance. transform() can work
	 * either
	 * in-place or out-of-place. The underlying data is shared and modified if
	 * supported is in-place mode.
	 *
	 * This class defines generic interface for transformers, fit() and
	 * transform().
	 * Subclasses should override fit(CFeatures*) or fit(CFeatures*, CLabels*)
	 * if necessary and transform() that transform transformation to features.
	 *
	 * Note that a new CFeatures is always created even in in-place mode because
	 * CFeatures is immutable.
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

		/** Fit transformer to features
		 * @param features the training features
		 */
		virtual void fit(CFeatures* features)
		{
		}

		/** Fit transformer to features and labels
		 * @param features the training features
		 * @param labels the training labels
		 */
		virtual void fit(CFeatures* features, CLabels* labels)
		{
			SG_SNOTIMPLEMENTED;
		}

		/** Apply transformation to features. If transformation is performed in
		 *  place, underlying data of input features will be reused if possible.
		 *	@param features features to transform
		 *	@param inplace whether transform in place
		 *	@return the result feature object after applying the transformer
		 */
		virtual CFeatures*
		transform(CFeatures* features, bool inplace = true) = 0;

		/** Apply inverse transformation to features. If transformation is
		 * performed in place, underlying data of input features will be reused
		 * if possible.
		 * @param features features to transform
		 * @param inplace whether transform in place
		 * @return the result feature object after inverse applying the
		 *transformer
		 */
		virtual CFeatures*
		inverse_transform(CFeatures* features, bool inplace = true)
		{
			SG_SNOTIMPLEMENTED;

			return nullptr;
		}
	};
}
#endif /* TRANSFORMER_H_ */
