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
	/** @brief Implements a reversible mapping from
	 * any form of labels to multi-class labels.
	 */
	class MulticlassLabelsEncoder : public LabelEncoder
	{
	public:
		MulticlassLabelsEncoder() = default;

		~MulticlassLabelsEncoder() = default;

		SGVector<float64_t> fit(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			if (!can_convert_float_to_int(result_vector))
			{
				std::set<float64_t> s(
				    result_vector.begin(), result_vector.end());
				io::warn(
				    "({}) have been converted to (0...{})", fmt::join(s, ", "),
				    result_vector.vlen - 1);
			}
			return fit_impl(result_vector);
		}

		std::shared_ptr<Labels>
		transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<MulticlassLabels>(
			    transform_impl(result_vector));
		}

		std::shared_ptr<Labels>
		inverse_transform(const std::shared_ptr<Labels>& labs) override
		{
			auto normalized_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<MulticlassLabels>(
			    inverse_transform_impl(normalized_vector));
		}

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

	protected:
		bool check_is_contiguous(const SGVector<float64_t>& vec) override
		{
			if (const auto vlen = unique_labels.size() == vec.size())
			{
				const auto [min_v, max_v] = std::minmax_element(
				    unique_labels.begin(), unique_labels.end());
				if (Math::fequals(*min_v, 0.0, eps) &&
				    Math::fequals(
				        *max_v, static_cast<float64_t>(vlen - 1), eps))
				{
					return true;
				}
			}
			return false;
		}
	};
} // namespace shogun

#endif