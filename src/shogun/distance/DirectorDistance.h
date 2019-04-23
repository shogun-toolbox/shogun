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
		virtual ~DirectorDistance()
		{
			cleanup();
		}

		virtual float64_t distance_function(int32_t x, int32_t y)
		{
			SG_ERROR("Distance function of Director Distance needs to be overridden.\n")
			return 0;
		}

		/** get distance function for lhs feature vector a
		  * and rhs feature vector b
		  *
		  * @param idx_a feature vector a at idx_a
		  * @param idx_b feature vector b at idx_b
		  * @return distance value
		 */
		virtual float64_t distance(int32_t idx_a, int32_t idx_b)
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
		virtual float64_t distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound)
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
		virtual bool init(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs)
		{
			if (this->parallel->get_num_threads()!=1)
			{
				SG_WARNING("Enforcing to use only one thread due to restrictions of directors\n")
				this->parallel->set_num_threads(1);
			}
			return Distance::init(lhs, rhs);
		}

		/** cleanup distance */
		virtual void cleanup()
		{

		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return Distance::get_num_vec_lhs();
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
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
		virtual bool has_features()
		{
			if (!external_features)
				return Distance::has_features();
			else
				return true;
		}

		/** remove lhs and rhs from distance */
		virtual void remove_lhs_and_rhs()
		{
			Distance::remove_lhs_and_rhs();
		}

		/// takes all necessary steps if the lhs is removed from distance matrix
		virtual void remove_lhs()
		{
			Distance::remove_lhs();
		}

		/// takes all necessary steps if the rhs is removed from distance matrix
		virtual void remove_rhs()
		{
			Distance::remove_rhs();
		}

		/** get distance type we are
		 *
		 * @return distance type DIRECTOR
		 */
		virtual EDistanceType get_distance_type() { return D_DIRECTOR; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type ANY
		 */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		/** get feature class the distance can deal with
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Director
		 */
		virtual const char* get_name() const { return "DirectorDistance"; }

		/** FIXME: precompute matrix should be dropped, handling
		 * should be via customdistance
		 *
		 * @param flag if precompute_matrix
		 */
		virtual void set_precompute_matrix(bool flag)
		{
			Distance::set_precompute_matrix(flag);
		}

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t x, int32_t y)
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
