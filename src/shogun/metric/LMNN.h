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

#ifdef HAVE_EIGEN3

#include <shogun/base/SGObject.h>
#include <shogun/distance/CustomMahalanobisDistance.h>
#include <shogun/features/Features.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>

#include <Eigen/Dense>

#include <set>
#include <vector>

namespace shogun
{

typedef std::set<int> ImpostorsSetType;
typedef std::vector< std::vector<Eigen::MatrixXd> > OuterProductsMatrixType;

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
		CLMNN(CFeatures* features, CMulticlassLabels* labels, uint32_t k);

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
		void train(SGMatrix<float64_t> init_transform);

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
		uint32_t get_k() const;

		/** set the number of target neighbours per example
		 *
		 * @param k the number of target neighbours per example
		 */
		void set_k(const uint32_t k);

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

		/** get maximum number of iterations
		 *
		 * @return maximum number of iterations
		 */
		uint32_t get_maxiter() const;

		/** set maximum number of iterations
		 *
		 * @param maximum number of iterations to set
		 */
		void set_maxiter(const uint32_t maxiter);

	private:
		/** register parameters */
		void init();

		/**
		 * for each feature in m_features, find its target neighbors; this is,
		 * its m_k nearest neighbors with the same label
		 */
		SGMatrix<int32_t> find_target_nn() const;

		/** for each pair of features in m_features, compute their outer product */
		OuterProductsMatrixType compute_outer_products() const;

		/** sum the outer products indexed by idxs in the matrix of outer products C */
		Eigen::MatrixXd sum_outer_products(const OuterProductsMatrixType& C, const SGMatrix<int32_t> idxs) const;

		/** find the impostors that remain after applying the transformation L */
		ImpostorsSetType find_impostors(const Eigen::MatrixXd& L, SGMatrix<int32_t> target_nn) const;

		/** update the gradient using the last transition in the impostors sets */
		void update_gradient(Eigen::MatrixXd& G, const OuterProductsMatrixType& C, const ImpostorsSetType& Nc, const ImpostorsSetType& Np) const;

		/** take gradient step and project onto positive semi-definite cone */
		void gradient_step(Eigen::MatrixXd& L, const Eigen::MatrixXd& G, float64_t stepsize) const;

		/** compute LMNN objective */
		float64_t compute_objective(const Eigen::MatrixXd& L, const OuterProductsMatrixType& C, const SGMatrix<int32_t> target_nn, const ImpostorsSetType& Nc) const;

	private:
		/** the linear transform learnt by LMNN once train has been called */
		SGMatrix<float64_t> m_linear_transform;

		/** training features */
		CFeatures* m_features;

		/** training labels */
		CLabels* m_labels;

		/**
		 * trade-off between pull and push forces in the objective; its default value
		 * is 0.5
		 */
		float64_t m_regularization;

		/** number of target neighbours to use per training example */
		uint32_t m_k;

		/** learning rate or step size used in gradient descent; its deafult value is 1e-07 */
		float64_t m_stepsize;

		/** maximum number of iterations; its default value is 1000 */
		uint32_t m_maxiter;

}; /* class CLMNN */

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

#endif /* LMNN_H_ */
