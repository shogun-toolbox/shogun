/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLSPLIT_H
#define SHOGUN_OPENMLSPLIT_H

#include <shogun/base/macros.h>
#include <shogun/features/Features.h>

#include <array>

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
		OpenMLSplit() : m_repeat_count(0), m_fold_count(0)
		{
		}

		OpenMLSplit(
		    const std::array<std::vector<int32_t>, 3>& train_idx,
		    const std::array<std::vector<int32_t>, 3>& test_idx)
		    : m_train_idx(train_idx), m_test_idx(test_idx)
		{
			// repeats and folds are zero indexed so add 1
			// we also assume that the repeats and folds indices go from
			// 0,1,...,N in increments of 1
			m_data_count =
			    std::make_pair(train_idx[0].size(), test_idx[0].size());
			m_repeat_count =
			    *std::max_element(train_idx[1].begin(), train_idx[1].end()) + 1;
			m_fold_count =
			    *std::max_element(train_idx[2].begin(), train_idx[2].end()) + 1;
			auto test_repeat_count =
			    *std::max_element(test_idx[1].begin(), test_idx[1].end()) + 1;
			auto test_fold_count =
			    *std::max_element(test_idx[2].begin(), test_idx[2].end()) + 1;

			REQUIRE(
			    train_idx[0].size() == train_idx[1].size() &&
			        train_idx[0].size() == train_idx[2].size(),
			    "All dimensions in train_idx must match!\n")
			REQUIRE(
			    test_idx[0].size() == test_idx[1].size() &&
			        test_idx[0].size() == test_idx[2].size(),
			    "All dimensions in test_idx must match!\n")

			if (m_repeat_count != test_repeat_count)
				SG_SERROR(
				    "Expected the train and test set to have the same number "
				    "of repeats, but got %d and %d respectively.\n",
				    m_repeat_count, test_repeat_count)
			if (m_repeat_count != test_repeat_count)
				SG_SERROR(
				    "Expected the train and test set to have the same number "
				    "of folds, but got %d and %d respectively.\n",
				    m_fold_count, test_fold_count)
		}

		static std::shared_ptr<OpenMLSplit>
		get_split(const std::string& split_url, const std::string& api_key);

		SG_FORCED_INLINE std::array<std::vector<int32_t>, 3>
		get_train_idx() const noexcept
		{
			return m_train_idx;
		}

		SG_FORCED_INLINE std::array<std::vector<int32_t>, 3>
		get_test_idx() const noexcept
		{
			return m_test_idx;
		}

		SG_FORCED_INLINE bool contains_splits() const noexcept
		{
			return !m_train_idx[0].empty() && !m_test_idx[0].empty();
		}

		SG_FORCED_INLINE int32_t get_num_repeats() const noexcept
		{
			return m_repeat_count;
		}

		SG_FORCED_INLINE int32_t get_num_folds() const noexcept
		{
			return m_fold_count;
		}

	private:
		static SGMatrix<float64_t>
		dense_feature_to_vector(const std::shared_ptr<CFeatures>& feat);

		static std::vector<OpenMLSplit::LabelType>
		nominal_feature_to_vector(const std::shared_ptr<CFeatures>& feat);

		std::array<std::vector<int32_t>, 3> m_train_idx;
		std::array<std::vector<int32_t>, 3> m_test_idx;
		std::pair<int32_t, int32_t> m_data_count;
		int32_t m_repeat_count;
		int32_t m_fold_count;
	};
} // namespace shogun
#endif // SHOGUN_OPENMLSPLIT_H
