/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#ifndef _MulticlassLabelsEncoder__H__
#define _MulticlassLabelsEncoder__H__

#include <memory>
#include <shogun/base/SGObject.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/LabelEncoder.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

	class MulticlassLabelsEncoder : public LabelEncoder
	{
	public:
		MulticlassLabelsEncoder() = default;

		~MulticlassLabelsEncoder() = default;

		/** Fit label encoder
		 *
		 * @param Target values.
		 * @return SGVector which contains unique labels.
		 */
		SGVector<float64_t> fit(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return fit_impl(result_vector);
		}
		/** Transform labels to normalized encoding.
		 *
		 * @param  Target values to be transformed.
		 * @return Labels transformed to be normalized.
		 */
		std::shared_ptr<Labels>
		transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<MulticlassLabels>(
			    transform_impl(result_vector));
		}
		/** Transform labels back to original encoding.
		 *
		 * @param normailzed encoding labels
		 * @return original encoding labels
		 */
		std::shared_ptr<Labels>
		inverse_transform(const std::shared_ptr<Labels>& labs) override
		{
			auto normalized_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<MulticlassLabels>(
			    inverse_transform_impl(normalized_vector));
		}
		/** Fit label encoder and return encoded labels.
		 *
		 * @param Target values.
		 * @return Labels transformed to be normalized.
		 */
		std::shared_ptr<Labels>
		fit_transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<MulticlassLabels>(
			    transform_impl(fit_impl(result_vector)));
		}

		virtual const char* get_name() const
		{
			return "MulticlassLabelsEncoder";
		}
	};
} // namespace shogun

#endif