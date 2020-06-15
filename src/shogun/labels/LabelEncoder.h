/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */

#ifndef _LABELENCODER__H__
#define _LABELENCODER__H__

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
namespace shogun
{

	class LabelEncoder : public SGObject
	{
	public:
		LabelEncoder() = default;

		virtual ~LabelEncoder() = default;

		/** Fit label encoder
		 *
		 * @param Target values.
		 * @return SGVector which contains unique labels.
		 */
		virtual SGVector<float64_t>
		fit(const std::shared_ptr<Labels>& labs) = 0;
		/** Transform labels to normalized encoding.
		 *
		 * @param  Target values to be transformed.
		 * @return Labels transformed to be normalized.
		 */
		virtual std::shared_ptr<Labels>
		transform(const std::shared_ptr<Labels>& labs) = 0;
		/** Transform labels back to original encoding.
		 *
		 * @param normailzed encoding labels
		 * @return original encoding labels
		 */
		virtual std::shared_ptr<Labels>
		inverse_transform(const std::shared_ptr<Labels>&) = 0;

		/** Fit label encoder and return encoded labels.
		 *
		 * @param Target values.
		 * @return Labels transformed to be normalized.
		 */
		virtual std::shared_ptr<Labels>
		fit_transform(const std::shared_ptr<Labels>&) = 0;

		virtual const char* get_name() const
		{
			return "LabelEncoder";
		}

	protected:
		SGVector<float64_t> fit_impl(const SGVector<float64_t>& origin_vector)
		{
			std::copy(
			    origin_vector.begin(), origin_vector.end(),
			    std::inserter(unique_labels, unique_labels.begin()));
			return SGVector<float64_t>(
			    unique_labels.begin(), unique_labels.end());
		}

		SGVector<float64_t>
		transform_impl(const SGVector<float64_t>& result_vector)
		{
			SGVector<float64_t> converted(result_vector.vlen);
			std::transform(
			    result_vector.begin(), result_vector.end(), converted.begin(),
			    [& unique_labels = unique_labels,
			     &normalized_to_origin =
			         normalized_to_origin](const auto& old_label) {
				    auto new_label = std::distance(
				        unique_labels.begin(), unique_labels.find(old_label));
				    normalized_to_origin[new_label] = old_label;
				    return new_label;
			    });
			return converted;
		}

		SGVector<float64_t>
		inverse_transform_impl(const SGVector<float64_t>& result_vector)
		{
			SGVector<float64_t> original_vector(result_vector.vlen);
			std::transform(
			    result_vector.begin(), result_vector.end(),
			    original_vector.begin(),
			    [& normalized_to_origin = normalized_to_origin](const auto& e) {
				    return normalized_to_origin[e];
			    });
			return original_vector;
		}
		std::set<float64_t> unique_labels;
		std::unordered_map<float64_t, float64_t> normalized_to_origin;
	};
} // namespace shogun

#endif