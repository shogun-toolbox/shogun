/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#ifndef _REGRESSIONLABELENCODER__H__
#define _REGRESSIONLABELENCODER__H__

#include <memory>
#include <shogun/base/SGObject.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/LabelEncoder.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/util/traits.h>

namespace shogun
{
	/** @brief Implements a reversible mapping from
	 * any form of labels to RegressionLabels.
	 */
	class RegressionLabelEncoder : public LabelEncoder
	{
	public:
		RegressionLabelEncoder() = default;

		~RegressionLabelEncoder() override = default;

		SGVector<float64_t> fit(const std::shared_ptr<Labels>& labs) override
		{
			return labs->as<DenseLabels>()->get_labels();
		}

		std::shared_ptr<Labels>
		transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<RegressionLabels>(result_vector);
		}
		
		std::shared_ptr<Labels>
		inverse_transform(const std::shared_ptr<Labels>& labs) override
		{
			auto normalized_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<RegressionLabels>(normalized_vector);
		}

		std::shared_ptr<Labels>
		fit_transform(const std::shared_ptr<Labels>& labs) override
		{
			const auto result_vector = labs->as<DenseLabels>()->get_labels();
			return std::make_shared<RegressionLabels>(result_vector);
		}

		const char* get_name() const override
		{
			return "RegressionLabelEncoder";
		}

	};
} // namespace shogun

#endif