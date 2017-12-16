/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Written (W) 2016 Soumyajit De
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _EUCLIDEANDISTANCE_H__
#define _EUCLIDEANDISTANCE_H__

#include <shogun/lib/config.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDotFeatures;
template <typename T> class CDenseFeatures;
template <typename T> class SGVector;

/** @brief class EuclideanDistance
 *
 * The familiar Euclidean distance for real valued features computes
 * the square root of the sum of squared disparity between the
 * corresponding feature dimensions of two data points.
 *
 * \f[\displaystyle
 *  d({\bf x},{\bf x'})= \sqrt{\sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|^2}
 * \f]
 *
 * This special case of Minkowski metric is invariant to an arbitrary
 * translation or rotation in feature space.
 *
 * The Euclidean Squared distance does not take the square root:
 *
 * \f[\displaystyle
 *  d({\bf x},{\bf x'})= \sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|^2
 * \f]
 *
 * Distance is computed as :
 *
 * \f[\displaystyle
 * \sqrt{{\bf x}\cdot {\bf x} - 2{\bf x}\cdot {\bf x'} + {\bf x'}\cdot {\bf x'}}
 * \f]
 *
 * Squared norms for left hand side and right hand side features can be precomputed.
 * WARNING : Make sure to reset squared norms using reset_squared_norms() when features
 * or feature matrix are changed.
 *
 * @see CMinkowskiMetric
 * @see <a href="http://en.wikipedia.org/wiki/Distance#Distance_in_Euclidean_space">
 * Wikipedia: Distance in Euclidean space</a>
 */
class CEuclideanDistance: public CDistance
{
public:
	/** default constructor */
	CEuclideanDistance();

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 */
	CEuclideanDistance(CDotFeatures* l, CDotFeatures* r);

	/** destructor */
	virtual ~CEuclideanDistance();

	/** init distance
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if init was successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/** cleanup distance */
	virtual void cleanup();

	/** get distance type we are
	 *
	 * @return distance type EUCLIDEAN
	 */
	virtual EDistanceType get_distance_type() { return D_EUCLIDEAN; }

	/** get feature class the distance can deal with
	 *
	 * @return feature class DENSE
	 */
	virtual EFeatureClass get_feature_class() { return C_ANY; }

	/** get feature type the distance can deal with
	 *
	 * @return feature type DREAL
	 */
	virtual EFeatureType get_feature_type() { return F_DREAL; }

	/** get name of the distance
	 *
	 * @return name Euclidean
	 */
	virtual const char* get_name() const { return "EuclideanDistance"; }

	/** disable application of sqrt on matrix computation
	 * the matrix can then also be named norm squared
	 *
	 * @return if application of sqrt is disabled
	 */
	virtual bool get_disable_sqrt() { return disable_sqrt; };

	/** disable application of sqrt on matrix computation
	 * the matrix can then also be named norm squared
	 *
	 * @param state new disable_sqrt
	 */
	virtual void set_disable_sqrt(bool state) { disable_sqrt=state; };

	/** compute the distance between lhs feature vector a
	 *  and rhs feature vector b. The computation of the
	 *  distance stops if the intermediate result is
	 *  larger than upper_bound. This is useful to use
	 *  with John Langford's Cover Tree
	 *
	 *  @param idx_a feature vector a at idx_a
	 *  @param idx_b feature vector b at idx_b
	 *  @param upper_bound value above which the computation
	 *  halts
	 *  @return distance value or upper_bound
	 */
	virtual float64_t distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound);

	/**
	 * Precomputation of squared norms for features of right hand side
	 * WARNING : Make sure to reset computations using reset_precompute()
	 * when features or feature matrix are changed.
	 */
	virtual void precompute_rhs();

	/**
	 * Precomputation of squared norms for features of left hand side
	 * WARNING : Make sure to reset computations using reset_precompute()
	 * when features or feature matrix are changed.
	 */
	virtual void precompute_lhs();

	/**
	 * Reset squared norm precomputations for features of both sides
	 * Should be used to reset whenever features or feature matrix are changed.
	 */
	virtual void reset_precompute();

	/** replace right-hand side features used in distance matrix
	 *
	 * make sure to check that your distance can deal with the
	 * supplied features (!)
	 *
	 * @param rhs features of right-hand side
	 * @return replaced right-hand side features
	 */
	virtual CFeatures* replace_rhs(CFeatures* rhs);

	/** replace left-hand side features used in distance matrix
	 *
	 * make sure to check that your distance can deal with the
	 * supplied features (!)
	 *
	 * @param lhs features of right-hand side
	 * @return replaced left-hand side features
	 */
	virtual CFeatures* replace_lhs(CFeatures* lhs);

	virtual SGMatrix<float64_t> get_distance_matrix();

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	template <class T> SGMatrix<float64_t> compute_distance_matrix(CDenseFeatures<T>* lhs, CDenseFeatures<T>* rhs);

	/** if application of sqrt on matrix computation is disabled */
	bool disable_sqrt;

	/** squared norms from features of right hand side */
	SGVector<float64_t> m_rhs_squared_norms;

	/** squared norms from features of left hand side */
	SGVector<float64_t> m_lhs_squared_norms;

private:
	/** initlaize by defaults and registers parameters */
	void register_params();

};

} // namespace shogun
#endif /* _EUCLIDEANDISTANCE_H__ */
