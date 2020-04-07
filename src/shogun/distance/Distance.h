/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Sergey Lisitsyn, 
 *          Saurabh Mahindre, Chiyuan Zhang, Michele Mazzoni, Evgeniy Andreev, 
 *          Evan Shelhamer, Yuyu Zhang, Heiko Strathmann, Thoralf Klein, 
 *          Bjoern Esser, Soumyajit De, Saurabh Goyal
 */

#ifndef _DISTANCE_H___
#define _DISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/Features.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
class File;
class Math;
class Features;

/** type of distance */
enum EDistanceType
{
	D_UNKNOWN = 0,
	D_MINKOWSKI = 10,
	D_MANHATTAN = 20,
	D_CANBERRA = 30,
	D_CHEBYSHEW = 40,
	D_GEODESIC = 50,
	D_JENSEN = 60,
	D_MANHATTANWORD = 70,
	D_HAMMINGWORD = 80,
	D_CANBERRAWORD = 90,
	D_SPARSEEUCLIDEAN = 100,
	D_EUCLIDEAN = 110,
	D_CHISQUARE = 120,
	D_TANIMOTO = 130,
	D_COSINE = 140,
	D_BRAYCURTIS = 150,
	D_CUSTOM = 160,
	D_ATTENUATEDEUCLIDEAN = 170,
	D_MAHALANOBIS = 180,
	D_DIRECTOR = 190,
	D_CUSTOMMAHALANOBIS = 200,
	D_LEVENSHTEIN = 210
};

/** @brief Class Distance, a base class for all the distances used in
 * the Shogun toolbox.
 *
 * The distance (or metric) is a function
 * \f$ d: X \times X \to R \f$ satisfying (for all \f$ x,y,z \in X\f$) conditions below:
 * - \f$ d(x,y) \geq 0 \f$
 *
 * - \f$ d(x,y) = 0\f$ if and only if \f$ x=y\f$
 *
 * - \f$ d(x,y) = d(y,x) \f$
 *
 * - \f$ d(x,y) \leq d(x,z) + d(z,y) \f$
 *
 * Currently distance inherited from the Distance class should be
 * symmetric.
 *
 * The simplest example of a distance function is the Euclidean
 * distance: @see EuclideanDistance
 *
 * In the means of Shogun toolbox the distance function is defined
 * on the 'space' of Features.
 *
 * Precomputations can be done for left hand side and right hand side features.
 * This has to be implemented in overloaded methods for precompute_lhs() and
 * precompute_rhs() in derived classes.
 * WARNING : Make sure to reset precomputations for features using reset_precompute()
 * when features or feature matrix are changed.
 *
 */
class Distance : public SGObject
{
	public:
		/** default constructor */
		Distance();

		/** init distance
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if init was successful
		 */
		Distance(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs);
		~Distance() override;

		/** get distance function for lhs feature vector a
		  * and rhs feature vector b
		  *
		  * @param idx_a feature vector a at idx_a
		  * @param idx_b feature vector b at idx_b
		  * @return distance value
		 */
		virtual float64_t distance(int32_t idx_a, int32_t idx_b);

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
			return distance(idx_a, idx_b);
		}

		/**
		 * Precomputation related to features of right hand side
		 * WARNING : Make sure to reset computations using reset_precompute()
		 * when features or feature matrix are changed.
		 * This method is empty, should be overloaded in derived class.
		 */
		virtual void precompute_rhs(){}

		/**
		 * Precomputation related to features of left hand side
		 * WARNING : Make sure to reset computations using reset_precompute()
		 * when features or feature matrix are changed.
		 * This method is empty, should be overloaded in derived class.
		 */
		virtual void precompute_lhs(){}

		/**
		 * Reset precomputations for features of both sides
		 * Should be used to reset whenever features or feature matrix are changed.
		 * This method is empty, should be overloaded in derived class.
		 */
		virtual void reset_precompute(){}

		/** get distance matrix
		 *
		 * @return computed distance matrix (needs to be cleaned up)
		 */
		SGMatrix<float64_t> get_distance_matrix()
		{
			return get_distance_matrix<float64_t>();
		}

		/** get distance matrix (templated)
		 *
		 * @return the distance matrix
		 */
		template <class T> SGMatrix<T> get_distance_matrix();

		/** compute row start offset for parallel kernel matrix computation
		 *
		 * @param offs offset
		 * @param n number of columns
		 * @param symmetric whether matrix is symmetric
		 */
		int32_t compute_row_start(int64_t offs, int32_t n, bool symmetric)
		{
			int32_t i_start;

			if (symmetric)
				i_start = (int32_t)Math::floor(
				    n - std::sqrt(Math::sq((float64_t)n) - offs));
			else
				i_start=(int32_t) (offs/int64_t(n));

			return i_start;
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
		virtual bool init(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs);

		/** cleanup distance
		 *
		 * abstract base method
		 */
		virtual void cleanup()=0;

		/** load the kernel matrix
		 *
		 * @param loader File object via which to load data
		 */
		void load(const std::shared_ptr<File>& loader);

		/** save kernel matrix
		 *
		 * @param writer File object via which to save data
		 */
		void save(const std::shared_ptr<File>& writer);

		/** get left-hand side features used in distance matrix
		 *
		 * @return left-hand side features
		 */
		inline std::shared_ptr<Features> get_lhs() {  return lhs; };

		/** get right-hand side features used in distance matrix
		 *
		 * @return right-hand side features
		 */
		inline std::shared_ptr<Features> get_rhs() {  return rhs; };

		/** replace right-hand side features used in distance matrix
		 *
		 * make sure to check that your distance can deal with the
		 * supplied features (!)
		 *
		 * @param rhs features of right-hand side
		 * @return replaced right-hand side features
		 */
		virtual std::shared_ptr<Features> replace_rhs(std::shared_ptr<Features> rhs);

		/** replace left-hand side features used in distance matrix
		 *
		 * make sure to check that your distance can deal with the
		 * supplied features (!)
		 *
		 * @param lhs features of right-hand side
		 * @return replaced left-hand side features
		 */
		virtual std::shared_ptr<Features> replace_lhs(std::shared_ptr<Features> lhs);

		/** remove lhs and rhs from distance */
		virtual void remove_lhs_and_rhs();

		/// takes all necessary steps if the lhs is removed from distance matrix
		virtual void remove_lhs();

		/// takes all necessary steps if the rhs is removed from distance matrix
		virtual void remove_rhs();

		/** get distance type we are
		 *
		 * abstrace base method
		 *
		 * @return distance type
		 */
		virtual EDistanceType get_distance_type()=0 ;

		/** get feature type the distance can deal with
		 *
		 * abstrace base method
		 *
		 * @return feature type
		 */
		virtual EFeatureType get_feature_type()=0;

		/** get feature class the distance can deal with
		 *
		 * abstract base method
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class()=0;

		/** FIXME: precompute matrix should be dropped, handling
		 * should be via customdistance
		 *
		 * @return if precompute_matrix
		 */
		inline bool get_precompute_matrix() { return precompute_matrix ;  }

		/** FIXME: precompute matrix should be dropped, handling
		 * should be via customdistance
		 *
		 * @param flag if precompute_matrix
		 */
		virtual void set_precompute_matrix(bool flag)
		{
			precompute_matrix=flag;

			if (!precompute_matrix)
			{
				SG_FREE(precomputed_matrix);
				precomputed_matrix=NULL;
			}
		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return num_lhs;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
		{
			return num_rhs;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			return lhs && rhs;
		}

		/** test whether features on lhs and rhs are the same
		 *
		 * @return true if features are the same
		 */
		inline bool lhs_equals_rhs()
		{
			return lhs==rhs;
		}

		/**
		 * Function for computing distance values
		 *
		 * @param result array of distance values
		 * @param idx_r_start iteration start value
		 * @param idx_start start
		 * @param idx_stop iteration end value
		 * @param idx_a feature vector a at idx_a
		 */
		void run_distance_rhs(SGVector<float64_t>& result, const index_t idx_r_start, index_t idx_start, const index_t idx_stop, const index_t idx_a);

		/**
		 * Function for computing distance values
		 *
		 * @param result array of distance values
		 * @param idx_r_start iteration start value
		 * @param idx_start start
		 * @param idx_stop iteration end value
		 * @param idx_b feature vector b at idx_b
		 */
		void run_distance_lhs(SGVector<float64_t>& result, const index_t idx_r_start, index_t idx_start, const index_t idx_stop, const index_t idx_b);

	protected:

		/// run distance thread
		static void* run_distance_thread(void* p);

		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)=0;

		/// matrix precomputation
		void do_precompute_matrix();

		/**
		 * Checks the compatibility between two supplied features
		 *
		 * @param l left hand side features
		 * @param r right hand side features
		 * @return true if the features are compatible
		 */
		virtual bool check_compatibility(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

	private:
		void init();

	protected:
		/** FIXME: precompute matrix should be dropped, handling
		 * should be via customdistance
		 */
		float32_t * precomputed_matrix;

		/** FIXME: precompute matrix should be dropped, handling
		 * should be via customdistance
		 */
		bool precompute_matrix;

		/// feature vectors to occur on the left hand side
		std::shared_ptr<Features> lhs;
		/// feature vectors to occur on the right hand side
		std::shared_ptr<Features> rhs;

		/** number of feature vectors on the left hand side */
		int32_t num_lhs;
		/** number of feature vectors on the right hand side */
		int32_t num_rhs;

};
} // namespace shogun
#endif
