/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __DATAGENERATOR_H_
#define __DATAGENERATOR_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun
{

/** @brief Class that is able to generate various data samples, which may be
 * used for examples in SHOGUN.
 */
class CDataGenerator: public CSGObject
{
public:
	CDataGenerator();

	virtual ~CDataGenerator();

	/** Generate points for classification tasks. Every dimension is in the range [0,1]. You can
	 * scale or translate the features afterwards. It works be creating a grid in the n-dimensional space
	 * with enough cells to hold all the classes and then starts filling them one by one.
	 *
	 * @param num_classes The number of classes
	 * @param dim the dimension of the target feature space
	 * @param num_points how many points to generate
	 * @param overlap controls whether classes can overlap. Takes values in [0,1], where 0 means no overlap permitted.
	 *
	 * @return a matrix of (dim+1)-dimension vector points, where vector[dim] corrresponds to the class of the point
	 */
	static SGMatrix<float64_t> generate_checkboard_data(int32_t num_classes,
		int32_t dim, int32_t num_points, float64_t overlap);

	/** Takes each m samples from two distributions p and q, where each element
	 * is standard normally distributed, except for the first dimension of q,
	 * where the mean is shifted by a specified value.
	 *
	 * May be used for a two-sample test.
	 *
	 * @param m number of samples to generate
	 * @param dim dimension of generated samples
	 * @param mean_shift is added to mean of first dimension
	 * @param target if non-empty then this is used as pre-allocated matrix.
	 * Make sure that its dimensions fit
	 * @return matrix with concatenated samples,first p then q
	 */
	static SGMatrix<float64_t> generate_mean_data(index_t m, index_t dim,
			float64_t mean_shift,
			SGMatrix<float64_t> target=SGMatrix<float64_t>());

	/** Produces samples as in source (g) from Table 3 in [1].
	 * Namely, produces an equal mixture of two independent Gaussians per --
	 * per dimension, of which there are two. The resulting 4 Gaussian blobs
	 * are then optionally rotated by the provided angle.
	 * Distance of means from origin (dimension-wise) can be controlled via
	 * parameter d
	 *
	 * May be used in a independence test to detect dependence in rotation.
	 * First dimensions can be used as one-dimensional p, second as q
	 *
	 * ﻿[1]: Gretton, A., Herbrich, R., Smola, A., Bousquet, O., & Schölkopf, B.
	 * (2005). Kernel Methods for Measuring Independence.
	 * Journal of Machine Learning Research, 6, 2075-2129.
	 *
	 * @param m number of samples per dimension
	 * @param d distance of Gaussian means to origin (dimension wise)
	 * @param angle fraction of \f$\pi\f$ that data is rotated by
	 * @param target if non-empty then this is used as pre-allocated matrix.
	 * Make sure that its dimensions fit
	 * @return TODO
	 */
	static SGMatrix<float64_t> generate_sym_mix_gauss(index_t m,
			float64_t d, float64_t angle,
			SGMatrix<float64_t> target=SGMatrix<float64_t>());

#ifdef HAVE_LAPACK
	/** Produces samples of gaussians
	 * The functions produces m number of samples of each gaussians (n number) with
	 * the given dimension.
	 *
	 * @param m number of samples
	 * @param n number of gaussians
	 * @param dim dimension of the multivariate normal distribution
	 * @return dim times (m*n) matrix with concatenated samples first m number
	 * of the first gaussian, m number of second etc.
	 */
	static SGMatrix<float64_t> generate_gaussians(index_t m, index_t n, index_t dim);
#endif /* HAVE_LAPACK */

	virtual const char* get_name() const { return "DataGenerator"; }

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

};

}

#endif /* __DATAGENERATOR_H_ */
