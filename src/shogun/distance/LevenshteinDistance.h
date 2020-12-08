/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#ifndef _LEVENSTEINDISTANCE_H__
#define _LEVENSTEINDISTANCE_H__

#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

namespace shogun
{

	class Features;
	class DotFeatures;
	template <typename T>
	class SGVector;

	class LevenshteinDistance : public Distance
	{
	public:
		/** default constructor */
		LevenshteinDistance();

		/** constructor
		 *
		 * @param lhs class name of left-hand side
		 * @param rhs class name of right-hand side
		 *
		 */
		LevenshteinDistance(
		    std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** destructor */
		~LevenshteinDistance() override
		{
		}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		bool
		init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** cleanup distance */
		void cleanup() override
		{
		}

		/** get distance type we are
		 *
		 * @return distance type D_LEVENSHTEIN
		 */
		EDistanceType get_distance_type() override
		{
			return D_LEVENSHTEIN;
		}

		/** get feature class the distance can deal with
		 *
		 * @return feature class STRING
		 */
		EFeatureClass get_feature_class() override
		{
			return C_STRING;
		}

		/** get feature type the distance can deal with
		 *
		 * @return feature type F_UNKNOWN
		 */
		EFeatureType get_feature_type() override
		{
			return F_UNKNOWN;
		}

		/** get name of the distance
		 *
		 * @return name Levenshtein
		 */
		const char* get_name() const override
		{
			return "LevenshteinDistance";
		}

		/** replace right-hand side features used in distance matrix
		 *
		 * make sure to check that your distance can deal with the
		 * supplied features (!)
		 *
		 * @param rhs features of right-hand side
		 * @return replaced right-hand side features
		 */
		std::shared_ptr<Features>
		replace_rhs(std::shared_ptr<Features> rhs) override;

		/** replace left-hand side features used in distance matrix
		 *
		 * make sure to check that your distance can deal with the
		 * supplied features (!)
		 *
		 * @param lhs features of right-hand side
		 * @return replaced left-hand side features
		 */

		std::shared_ptr<Features>
		replace_lhs(std::shared_ptr<Features> lhs) override;
		
		using Distance::distance;

		int32_t distance(const std::string& lhs, const std::string& rhs)
		{
			const SGVector<char> l(lhs.begin(), lhs.end());
			const SGVector<char> r(rhs.begin(), rhs.end());
			return compute_impl(l, r);
		}
	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	private:
		float64_t
		compute_impl(const SGVector<char>& lhs, const SGVector<char>& rhs);
	};

} // namespace shogun

#endif /* _LEVENSTEINDISTANCE_H__ */