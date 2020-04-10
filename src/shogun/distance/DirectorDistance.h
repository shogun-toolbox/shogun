/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang, Chiyuan Zhang,
 *          Viktor Gal, Sergey Lisitsyn, Bjoern Esser
 */

#ifndef _DIRECTORDISTANCE_H___
#define _DIRECTORDISTANCE_H___

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS

#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class DirectorDistance : public Distance
{
	public:
		/* default constructor */
		DirectorDistance(bool is_external_features)
		: Distance(), external_features(is_external_features)
		{

		}

		/** destructor */
		~DirectorDistance() override
		{
			cleanup();
		}

		virtual float64_t distance_function(int32_t x, int32_t y)
		{
			error("Distance function of Director Distance needs to be overridden.");
			return 0;
		}

		/** get distance function for lhs feature vector a
		  * and rhs feature vector b
		  *
		  * @param idx_a feature vector a at idx_a
		  * @param idx_b feature vector b at idx_b
		  * @return distance value
		 */
		float64_t distance(int32_t idx_a, int32_t idx_b) override
		{
			if (idx_a < 0 || idx_b <0)
				return 0;

			if (!external_features)
				Distance::distance(idx_a, idx_b);
			else
				return compute(idx_a, idx_b);
		}

		/** get distance function for lhs feature vector a
		 *  and rhs feature vector b. The computation of the
		 *  distance stops if the intermediate result is
		 *  larger than upper_bound. This is useful to use
		 *  with John Langford's Cover Tree and it is ONLY
		 *  implemented for Euclidean distance
		 *
		 *  @param idx_a feature vector a at idx_a
		 *  @param idx_b feature vector b at idx_b
		 *  @param upper_bound value above which the computation
		 *  halts
		 *  @return distance value or upper_bound
		 */
		float64_t distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound) override
		{
			return Distance::distance(idx_a, idx_b);
		}

		/** init distance
		 *
		 *  make sure to check that your distance can deal with the
		 *  supplied features (!)
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if init was successful
		 */
		bool init(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs) override
		{
			if (env()->get_num_threads()!=1)
			{
				io::warn("Enforcing to use only one thread due to restrictions of directors");
				env()->set_num_threads(1);
			}
			return Distance::init(lhs, rhs);
		}

		/** cleanup distance */
		void cleanup() override
		{

		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		int32_t get_num_vec_lhs() override
		{
			return Distance::get_num_vec_lhs();
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		int32_t get_num_vec_rhs() override
		{
			return Distance::get_num_vec_rhs();
		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual void set_num_vec_lhs(int32_t num)
		{
			num_lhs=num;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual void set_num_vec_rhs(int32_t num)
		{
			num_rhs=num;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		bool has_features() override
		{
			if (!external_features)
				return Distance::has_features();
			else
				return true;
		}

		/** remove lhs and rhs from distance */
		void remove_lhs_and_rhs() override
		{
			Distance::remove_lhs_and_rhs();
		}

		/// takes all necessary steps if the lhs is removed from distance matrix
		void remove_lhs() override
		{
			Distance::remove_lhs();
		}

		/// takes all necessary steps if the rhs is removed from distance matrix
		void remove_rhs() override
		{
			Distance::remove_rhs();
		}

		/** get distance type we are
		 *
		 * @return distance type DIRECTOR
		 */
		EDistanceType get_distance_type() override { return D_DIRECTOR; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type ANY
		 */
		EFeatureType get_feature_type() override { return F_ANY; }

		/** get feature class the distance can deal with
		 *
		 * @return feature class ANY
		 */
		EFeatureClass get_feature_class() override { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Director
		 */
		const char* get_name() const override { return "DirectorDistance"; }

		/** FIXME: precompute matrix should be dropped, handling
		 * should be via customdistance
		 *
		 * @param flag if precompute_matrix
		 */
		void set_precompute_matrix(bool flag) override
		{
			Distance::set_precompute_matrix(flag);
		}

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t x, int32_t y) override
		{
			return distance_function(x, y);
		}

	protected:
		/* */
		bool external_features;
};

}

#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTORDISTANCE_H___ */
