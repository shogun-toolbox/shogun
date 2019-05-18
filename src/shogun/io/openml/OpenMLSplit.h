/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLSPLIT_H
#define SHOGUN_OPENMLSPLIT_H

#include <shogun/base/macros.h>
#include <shogun/features/Features.h>

namespace shogun
{
	/**
	 * Handles an OpenML split.
	 */
	class OpenMLSplit
	{
	public:
		enum class LabelType
		{
			TRAIN = 1,
			TEST = 2
		};

		/**
		 * Default constructor. This is used when there are no
		 * train or test indices.
		 */
		OpenMLSplit() = default;

		OpenMLSplit(
		    std::vector<std::vector<int64_t>> train_idx,
		    std::vector<std::vector<int64_t>> test_idx)
		    : m_train_idx(std::move(train_idx)), m_test_idx(std::move(test_idx))
		{
		}

		static std::shared_ptr<OpenMLSplit>
		get_split(const std::string& split_url, const std::string& api_key);

		SG_FORCED_INLINE std::vector<std::vector<int64_t>> get_train_idx() const
		    noexcept
		{
			return m_train_idx;
		}

		SG_FORCED_INLINE std::vector<std::vector<int64_t>> get_test_idx() const
		    noexcept
		{
			return m_test_idx;
		}

		SG_FORCED_INLINE bool contains_splits() const noexcept
		{
			return !m_train_idx.empty() && !m_test_idx.empty();
		}

	private:
		static SGVector<float64_t>
		dense_feature_to_vector(const std::shared_ptr<CFeatures>& feat);

		static std::vector<OpenMLSplit::LabelType>
		string_feature_to_vector(const std::shared_ptr<CFeatures>& feat);

		std::vector<std::vector<int64_t>> m_train_idx;
		std::vector<std::vector<int64_t>> m_test_idx;
	};
} // namespace shogun
#endif // SHOGUN_OPENMLSPLIT_H
