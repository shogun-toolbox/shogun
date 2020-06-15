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
			fit_impl(result_vector);
			require(
			    unique_labels.size() == 2,
			    "BinaryLabel should contain only two elements");

			return SGVector<float64_t>(
			    unique_labels.begin(), unique_labels.end());
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
			    std::set<float64_t>(result_vector.begin(), result_vector.end())
			            .size() == 2,
			    "BinaryLabel should contain only two elements");
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
			auto normalized_vector = labs->as<DenseLabels>()->get_labels();

			std::transform(
			    normalized_vector.begin(), normalized_vector.end(),
			    normalized_vector.begin(), [](float64_t e) {
				    if (std::abs(e + 1.0) <=
				        std::numeric_limits<float64_t>::epsilon())
					    return 0.0;
				    else
					    return e;
			    });
			require(
			    std::set<float64_t>(
			        normalized_vector.begin(), normalized_vector.end())
			            .size() == 2,
			    "BinaryLabel should contain only two elements");

			return std::make_shared<BinaryLabels>(
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