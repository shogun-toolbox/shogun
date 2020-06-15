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

	class BinaryLabelEncoder : public LabelEncoder
	{
	public:
		BinaryLabelEncoder() = default;

		~BinaryLabelEncoder() = default;

		/** Fit label encoder
		 *
		 * @param Target values.
		 * @return SGVector which contains unique labels.
		 */
		SGVector<float64_t> fit(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			auto result_labels = fit_impl(result_vector);
			require(
			    unique_labels.size() == 2,
			    "Binary Labels should contain only two elements");

			return result_labels;
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
			require(
			    std::unordered_set<float64_t>(
			        result_vector.begin(), result_vector.end())
			            .size() == 2,
			    "Binary Labels should contain only two elements");
			auto transformed_vec = transform_impl(result_vector);

			std::transform(
			    transformed_vec.begin(), transformed_vec.end(),
			    transformed_vec.begin(), [](float64_t e) {
				    if (std::abs(e - 0.0) <=
				        std::numeric_limits<float64_t>::epsilon())
					    return -1.0;
				    else
					    return e;
			    });
			return std::make_shared<BinaryLabels>(transformed_vec);
		}
		/** Transform labels back to original encoding.
		 *
		 * @param normailzed encoding labels
		 * @return original encoding labels
		 */
		std::shared_ptr<Labels>
		inverse_transform(const std::shared_ptr<Labels>& labs) override
		{
			auto normalized_labels = labs->as<BinaryLabels>();
			normalized_labels->ensure_valid();
			auto normalized_vector = normalized_labels->get_labels();
			require(
			    std::unordered_set<float64_t>(
			        normalized_vector.begin(), normalized_vector.end())
			            .size() == 2,
			    "Binary Labels should contain only two elements");

			std::transform(
			    normalized_vector.begin(), normalized_vector.end(),
			    normalized_vector.begin(), [](float64_t e) {
				    if (std::abs(e + 1.0) <=
				        std::numeric_limits<float64_t>::epsilon())
					    return 0.0;
				    else
					    return e;
			    });
			auto origin_vec = inverse_transform_impl(normalized_vector);
			SGVector<int32_t> result_vev(origin_vec.vlen);
			std::transform(
			    origin_vec.begin(), origin_vec.end(), result_vev.begin(),
			    [](auto&& e) { return static_cast<int32_t>(e); });
			return std::make_shared<BinaryLabels>(result_vev);
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
			return std::make_shared<BinaryLabels>(
			    transform_impl(fit_impl(result_vector)));
		}

		virtual const char* get_name() const
		{
			return "BinaryLabelEncoder";
		}
	};
} // namespace shogun

#endif