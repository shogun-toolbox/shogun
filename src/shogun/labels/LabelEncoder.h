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
	/** @brief Implements a reversible mapping from any
	 * form of labels to one of Shogun's target label spaces
	 * (binary, multi-class, etc).
	 */
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
		virtual bool check_is_contiguous(
		    const SGVector<float64_t>& vec, const std::set<float64_t>& labels)
		{
			return false;
		}
		SGVector<float64_t> fit_impl(const SGVector<float64_t>& origin_vector)
		{
			is_fitted = true;
			std::copy(
			    origin_vector.begin(), origin_vector.end(),
			    std::inserter(unique_labels, unique_labels.begin()));
			if (check_is_contiguous(origin_vector, unique_labels))
			{
				is_fitted = false;
			}
			return SGVector<float64_t>(
			    unique_labels.begin(), unique_labels.end());
		}

		SGVector<float64_t>
		transform_impl(const SGVector<float64_t>& result_vector)
		{
			is_transformed = true;
			if (!is_fitted && unique_labels.size())
				return result_vector;
			require(is_fitted, "Transform expect to be called after fit.");
			SGVector<float64_t> converted(result_vector.vlen);
			std::transform(
			    result_vector.begin(), result_vector.end(), converted.begin(),
			    [& unique_labels = unique_labels,
			     &inverse_mapping = inverse_mapping](const auto& old_label) {
				    auto new_label = std::distance(
				        unique_labels.begin(), unique_labels.find(old_label));
				    inverse_mapping[new_label] = old_label;
				    return new_label;
			    });
			return converted;
		}

		SGVector<float64_t>
		inverse_transform_impl(const SGVector<float64_t>& result_vector)
		{
			require(
			    is_transformed,
			    "Inverse transform expect to be called after transform.");
			if (!is_fitted && unique_labels.size() && is_transformed)
			{
				return result_vector;
			}
			SGVector<float64_t> original_vector(result_vector.vlen);
			std::transform(
			    result_vector.begin(), result_vector.end(),
			    original_vector.begin(),
			    [& inverse_mapping = inverse_mapping](const auto& e) {
				    return inverse_mapping[e];
			    });
			return original_vector;
		}

		bool can_convert_float_to_int(const SGVector<float64_t>& vec) const
		{
			SGVector<int32_t> converted(vec.vlen);
			std::transform(
			    vec.begin(), vec.end(), converted.begin(),
			    [](auto&& e) { return static_cast<int32_t>(e); });
			return std::equal(
			    vec.begin(), vec.end(), converted.begin(),
			    [&](auto&& e1, auto&& e2) {
				    return Math::fequals(e1, static_cast<float64_t>(e2), eps);
			    });
		}

		std::set<float64_t> unique_labels;
		std::unordered_map<float64_t, float64_t> inverse_mapping;
		const float64_t eps = std::numeric_limits<float64_t>::epsilon();
		bool is_fitted = false;
		bool is_transformed = false;
	};
} // namespace shogun

#endif