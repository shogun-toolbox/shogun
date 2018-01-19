/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jiaolong Xu, Soeren Sonnenburg, Heiko Strathmann, Evgeniy Andreev, 
 *          Yuyu Zhang, Fernando Iglesias, Björn Esser
 */

#ifndef _LABELS_FACTORY__H__
#define _LABELS_FACTORY__H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{
	class SGObject;
	class CLabels;
	class CBinaryLabels;
	class CLatentLabels;
	class CMulticlassLabels;
	class CRegressionLabels;
	class CStructuredLabels;
	class CMultilabelLabels;
	class CMulticlassSOLabels;

/** @brief The helper class to specialize base class instances of labels
 */
class CLabelsFactory : public CSGObject
{
public:
	/** specialize a base class instance to CBinaryLabels
	 *
	 * @param base_labels its dynamic type must be CBinaryLabels
	 */
	static CBinaryLabels* to_binary(CLabels* base_labels);

	/** specialize a base class instance to CLatentLabels
	 *
	 * @param base_labels its dynamic type must be CLatentLabels
	 */
	static CLatentLabels* to_latent(CLabels* base_labels);

	/** specialize a base class instance to CMulticlassLabels
	 *
	 * @param base_labels its dynamic type must be CMulticlassLabels
	 */
	static CMulticlassLabels* to_multiclass(CLabels* base_labels);

	/** specialize a base class instance to CRegressionLabels
	 *
	 * @param base_labels its dynamic type must be CRegressionLabels
	 */
	static CRegressionLabels* to_regression(CLabels* base_labels);

	/** specialize a base class instance to CStructuredLabels
	 *
	 * @param base_labels its dynamic type must be CStructuredLabels
	 */
	static CStructuredLabels* to_structured(CLabels* base_labels);

	/** specialize a base class instance to CMultilabelLabels
	 *
	 * @param base_labels its dynamic type must be CMultilabelLabels
	 */
	static CMultilabelLabels* to_multilabel_output(CLabels* base_labels);

	/** specialize a base class instance to CMulticlassSOLabels
	 *
	 * @param base_labels its dynamic type must be CMulticlassSOLabels
	 */
	static CMulticlassSOLabels* to_multiclass_structured(CLabels* base_labels);

	/** @return object name */
	virtual const char* get_name() const { return "LabelsFactory"; }
};

}

#endif /* _LABELS_FACTORY__H__ */
