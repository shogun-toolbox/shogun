/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTANCE_H___
#define _DISTANCE_H___

#include <stdio.h>

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/Features.h>

namespace shogun
{
class CFile;
class CMath;
class CFeatures;

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
	D_HAMMINGWORD = 80 ,
	D_CANBERRAWORD = 90,
	D_SPARSEEUCLIDIAN = 100,
	D_EUCLIDIAN = 110,
	D_CHISQUARE = 120,
	D_TANIMOTO = 130,
	D_COSINE = 140,
	D_BRAYCURTIS = 150,
	D_CUSTOM = 160,
	D_ATTENUATEDEUCLIDIAN = 170,
        D_MAHALANOBIS = 180
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
 * Currently distance inherited from the CDistance class should be
 * symmetric.
 *
 * The simpliest example of a distance function is the euclidian
 * distance: @see CEuclidianDistance
 *
 * In the means of Shogun toolbox the distance function is defined
 * on the 'space' of CFeatures.
 *
 */
class CDistance : public CSGObject
{
	public:
		/** default constructor */
		CDistance();

		/** init distance
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if init was successful
		 */
		CDistance(CFeatures* lhs, CFeatures* rhs);
		virtual ~CDistance();

		/** get distance function for lhs feature vector a
		  * and rhs feature vector b
		  *
		  * @param idx_a feature vector a at idx_a
		  * @param idx_b feature vector b at idx_b
		  * @return distance value
		 */
		inline float64_t distance(int32_t idx_a, int32_t idx_b)
		{
			if (idx_a < 0 || idx_b <0)
				return 0;

			ASSERT(lhs);
			ASSERT(rhs);

			if (lhs==rhs)
			{
				int32_t num_vectors = lhs->get_num_vectors();

				if (idx_a>=num_vectors)
					idx_a=2*num_vectors-1-idx_a;

				if (idx_b>=num_vectors)
					idx_b=2*num_vectors-1-idx_b;
			}

			ASSERT(idx_a<lhs->get_num_vectors());
			ASSERT(idx_b<rhs->get_num_vectors());

			if (precompute_matrix && (precomputed_matrix==NULL) && (lhs==rhs))
				do_precompute_matrix() ;

			if (precompute_matrix && (precomputed_matrix!=NULL))
			{
				if (idx_a>=idx_b)
					return precomputed_matrix[idx_a*(idx_a+1)/2+idx_b] ;
				else
					return precomputed_matrix[idx_b*(idx_b+1)/2+idx_a] ;
			}

			return compute(idx_a, idx_b);
		}

		/** get distance matrix
		 *
		 * @return computed distance matrix (needs to be cleaned up)
		 */
		SGMatrix<float64_t> get_distance_matrix();

		/** get distance matrix real
		 *
		 * @param m dimension m
		 * @param n dimension n
		 * @param target target matrix
		 * @return target matrix
		 */
		virtual float64_t* get_distance_matrix_real(
			int32_t &m,int32_t &n, float64_t* target);

		/** get distance matrix short real
		 *
		 * @param m dimension m
		 * @param n dimension n
		 * @param target target matrix
		 * @return target matrix
		 */
		virtual float32_t* get_distance_matrix_shortreal(
			int32_t &m,int32_t &n,float32_t* target);

		/** init distance
		 *
		 *  make sure to check that your distance can deal with the
		 *  supplied features (!)
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* lhs, CFeatures* rhs);

		/** cleanup distance
		 *
		 * abstract base method
		 */
		virtual void cleanup()=0;

		/** load the kernel matrix
		 *
		 * @param loader File object via which to load data
		 */
		void load(CFile* loader);

		/** save kernel matrix
		 *
		 * @param writer File object via which to save data
		 */
		void save(CFile* writer);

		/** get left-hand side features used in distance matrix
		 *
		 * @return left-hand side features
		 */
		inline CFeatures* get_lhs() { SG_REF(lhs); return lhs; };

		/** get right-hand side features used in distance matrix
		 *
		 * @return right-hand side features
		 */
		inline CFeatures* get_rhs() { SG_REF(rhs); return rhs; };
    
		/** replace right-hand side features used in distance matrix
		 *
		 * make sure to check that your distance can deal with the
		 * supplied features (!)
		 *
		 * @param rhs features of right-hand side
		 * @return replaced right-hand side features
		 */
		CFeatures* replace_rhs(CFeatures* rhs);
            
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
		inline virtual void set_precompute_matrix(bool flag)
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
		inline int32_t get_num_vec_lhs()
		{
			if (!lhs)
				return 0;
			else
				return lhs->get_num_vectors();
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		inline int32_t get_num_vec_rhs()
		{
			if (!rhs)
				return 0;
			else
				return rhs->get_num_vectors();
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		inline bool has_features()
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

	protected:

		/// run distance thread
		static void* run_distance_thread(void* p);		

		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t x, int32_t y)=0;

		/// matrix precomputation
		void do_precompute_matrix();

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

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;

};
} // namespace shogun
#endif
