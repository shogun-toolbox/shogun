/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifndef LMNN_H_
#define LMNN_H_

#include <lib/config.h>

#ifdef HAVE_EIGEN3
#ifdef HAVE_LAPACK

#include <base/SGObject.h>
#include <distance/CustomMahalanobisDistance.h>
#include <features/DenseFeatures.h>
#include <labels/MulticlassLabels.h>
#include <lib/SGMatrix.h>

namespace shogun
{

// Forward declaration
class CLMNNStatistics;

/**
 * @brief Class LMNN that implements the distance metric learning technique
 * Large Margin Nearest Neighbour (LMNN) described in
 *
 * Weinberger, K. Q., Saul, L. K.
 * Distance Metric Learning for Large Margin Nearest Neighbor Classification.
 */
class CLMNN : public CSGObject
{
	public:
		/** default constructor */
		CLMNN();

		/** standard constructor
		 *
		 * @param features feature vectors
		 * @param labels labels of the features
		 * @param k number of target neighbours per example
		 */
		CLMNN(CDenseFeatures<float64_t>* features, CMulticlassLabels* labels, int32_t k);

		/** destructor */
		virtual ~CLMNN();

		/** @return name of SGSerializable */
		virtual const char* get_name() const;

		/**
		 * LMNN algorithm to learn a linear transformation of the original feature
		 * space (or, equivalently, a Mahalanobis distance) such that kNN
		 * classification performance is maximized
		 *
		 * @param init_transform initial linear transform
		 */
		void train(SGMatrix<float64_t> init_transform=SGMatrix<float64_t>());

		/** get the learnt linear transform (denoted L in LMNN literature typically)
		 *
		 * @return the linear transform L
		 */
		SGMatrix<float64_t> get_linear_transform() const;

		/**
		 * get the learnt Mahalanobis distance (typically denoted M in LMNN literature)
		 * encapsulated in a CCustomMahalanobisDistance object, suitable to be used in kNN
		 *
		 * @return the distance M
		 */
		CCustomMahalanobisDistance* get_distance() const;

		/** get the number of target neighbours per example
		 *
		 * @return number of neighbours per example
		 */
		int32_t get_k() const;

		/** set the number of target neighbours per example
		 *
		 * @param k the number of target neighbours per example
		 */
		void set_k(const int32_t k);

		/** get regularization
		 *
		 * @return regularization strength
		 */
		float64_t get_regularization() const;

		/** set regularization
		 *
		 * @param regularization regularization strength to set
		 */
		void set_regularization(const float64_t regularization);

		/** get step size
		 *
		 * @return step size
		 */
		float64_t get_stepsize() const;

		/** set step size
		 *
		 * @param stepsize step size to set
		 */
		void set_stepsize(const float64_t stepsize);

		/** get step size threshold
		 *
		 * @return step size threshold
		 */
		float64_t get_stepsize_threshold() const;

		/** set step size threshold
		 *
		 * @param stepsize_threshold step size threshold to set
		 */
		void set_stepsize_threshold(const float64_t stepsize_threshold);

		/** get maximum number of iterations
		 *
		 * @return maximum number of iterations
		 */
		uint32_t get_maxiter() const;

		/** set maximum number of iterations
		 *
		 * @param maxiter maximum number of iterations to set
		 */
		void set_maxiter(const uint32_t maxiter);

		/** get number of iterations between exact impostors search
		 *
		 * @return iterations between exact impostors search
		 */
		uint32_t get_correction() const;

		/** set number of iterations between exact impostors search
		 *
		 * @param correction iterations between exact impostors search
		 */
		void set_correction(const uint32_t correction);

		/** get objective threshold
		 *
		 * @return objective threshold
		 */
		float64_t get_obj_threshold() const;

		/** set objective threshold
		 *
		 * @param obj_threshold objective threshold to set
		 */
		void set_obj_threshold(const float64_t obj_threshold);

		/** get whether the linear transform will be diagonal
		 *
		 * @return whether the linear transform will be diagonal
		 */
		bool get_diagonal() const;

		/** set whether the linear transform will be diagonal
		 *
		 * @param diagonal whether the linear transform will be diagonal
		 */
		void set_diagonal(const bool diagonal);

		/** get LMNN training statistics
		 *
		 * @return LMNN training statistics
		 */
		CLMNNStatistics* get_statistics() const;

	private:
		/** register parameters */
		void init();

	private:
		/** the linear transform learnt by LMNN once train has been called */
		SGMatrix<float64_t> m_linear_transform;

		/** training features */
		CFeatures* m_features;

		/** training labels */
		CLabels* m_labels;

		/**
		 * trade-off between pull and push forces in the objective.
		 * Its default value is 0.5
		 */
		float64_t m_regularization;

		/** number of target neighbours to use per training example */
		int32_t m_k;

		/**
		 * learning rate or step size used in gradient descent.
		 * Its deafult value is 1e-07.
		 */
		float64_t m_stepsize;

		/**
		 * step size threshold; during training the step size is modified
		 * internally, stop training if the step size is below this threshold.
		 * Its default value is 1e-22.
		 */
		float64_t m_stepsize_threshold;

		/** maximum number of iterations. Its default value is 1000. */
		uint32_t m_maxiter;

		/**
		 * number of iterations between exact computation of impostors.
		 * Its default value is 15
		 */
		uint32_t m_correction;

		/**
		 * objective threshold; stop training if the first order difference in
		 * absolute value of the objective function in the last three iterations
		 * is below (element-wise) this threshold times the current objective.
		 * Its default value is 1e-9.
		 */
		float64_t m_obj_threshold;

		/**
		 * whether m_linear_transform is forced to be diagonal (useful to
		 * perform feature selection). Its default value is false.
		 */
		bool m_diagonal;

		/** training statistics, @see CLMNNStatistics */
		CLMNNStatistics* m_statistics;

}; /* class CLMNN */

/**
 * @brief Class LMNNStatistics used to give access to intermediate results
 * obtained training LMNN.
 */
class CLMNNStatistics : public CSGObject
{
	public:
		/** default constructor */
		CLMNNStatistics();

		/** destructor */
		virtual ~CLMNNStatistics();

		/** @return name of SGSerializable */
		virtual const char* get_name() const;

		/**
		 * resize CLMNNStatistics::obj, CLMNNStatistics::stepsize and
		 * CLMNNStatistics::num_impostors to fit the specified number of elements
		 *
		 * @param size number of elements
		 */
		void resize(int32_t size);

		/**
		 * set objective, step size and number of impostors computed at the
		 * specified iteration
		 *
		 * @param iter index to store the parameters, must be greater or equal to zero,
		 * and less than the size
		 * @param obj_iter objective to set
		 * @param stepsize_iter stepsize to set
		 * @param num_impostors_iter number of impostors to set
		 */
		void set(index_t iter, float64_t obj_iter, float64_t stepsize_iter, uint32_t num_impostors_iter);

	private:
		/** register parameters */
		void init();

	public:
		/** objective function at each iteration */
		SGVector<float64_t> obj;

		/** step size at each iteration */
		SGVector<float64_t> stepsize;

		/** number of impostors at each iteration */
		SGVector<uint32_t> num_impostors;
};

} /* namespace shogun */

#endif /* HAVE_LAPACK */
#endif /* HAVE_EIGEN3 */

#endif /* LMNN_H_ */
