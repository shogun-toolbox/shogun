/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#ifndef _BINARYLABELENCODER__H__
#define _BINARYLABELENCODER__H__

#include <memory>
#include <shogun/base/SGObject.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/LabelEncoder.h>
#include <shogun/lib/SGVector.h>
#include <unordered_set>
namespace shogun
{
	/** @brief Implements a reversible mapping from
	 * any form of labels to binary labels (+1, -1).
	 */
	class BinaryLabelEncoder : public LabelEncoder
	{
	public:
		BinaryLabelEncoder() = default;

		~BinaryLabelEncoder() override = default;

		SGVector<float64_t> fit(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			check_is_valid(result_vector);
			if (!can_convert_float_to_int(result_vector))
			{
				std::set<float64_t> s(
				    result_vector.begin(), result_vector.end());
				io::warn(
				    "({}, {}) have been converted to (-1, 1).", *s.begin(),
				    *s.end());
			}
			return fit_impl(result_vector);
		}

		std::shared_ptr<Labels>
		transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			check_is_valid(result_vector);
			auto transformed_vec = transform_impl(result_vector);

			std::transform(
			    transformed_vec.begin(), transformed_vec.end(),
			    transformed_vec.begin(), [](float64_t e) {
				    return Math::fequals(
				               e, 0.0,
				               std::numeric_limits<float64_t>::epsilon())
				               ? -1.0
				               : e;
			    });
			return std::make_shared<BinaryLabels>(transformed_vec);
		}

		std::shared_ptr<Labels>
		inverse_transform(const std::shared_ptr<Labels>& labs) override
		{
			auto normalized_labels = labs->as<BinaryLabels>();
			normalized_labels->ensure_valid();
			auto normalized_vector = normalized_labels->get_labels();
			std::transform(
			    normalized_vector.begin(), normalized_vector.end(),
			    normalized_vector.begin(), [](float64_t e) {
				    return Math::fequals(
				               e, -1.0,
				               std::numeric_limits<float64_t>::epsilon())
				               ? 0.0
				               : e;
			    });
			auto origin_vec = inverse_transform_impl(normalized_vector);
			SGVector<int32_t> result_vev(origin_vec.vlen);
			std::transform(
			    origin_vec.begin(), origin_vec.end(), result_vev.begin(),
			    [](auto&& e) { return static_cast<int32_t>(e); });
			return std::make_shared<BinaryLabels>(result_vev);
		}

		std::shared_ptr<Labels>
		fit_transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<BinaryLabels>(
			    transform_impl(fit_impl(result_vector)));
		}

		const char* get_name() const override
		{
			return "BinaryLabelEncoder";
		}

	private:
		void check_is_valid(const SGVector<float64_t>& vec)
		{
			const auto unique_set =
			    std::unordered_set<float64_t>(vec.begin(), vec.end());
			require(
			    unique_set.size() == 2,
			    "Cannot interpret ({}) as binary labels, need exactly two "
			    "classes.",
			    fmt::join(unique_set, ", "));
		}
	};
} // namespace shogun

#endif