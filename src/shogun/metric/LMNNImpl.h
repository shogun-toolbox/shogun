/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifndef LMNNIMPL_H_
#define LMNNIMPL_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <Eigen/Dense>

#include <set>
#include <vector>

namespace shogun
{

class CImpostorNode;

typedef std::set<CImpostorNode> ImpostorsSetType;
typedef std::vector< std::vector<Eigen::MatrixXd> > OuterProductsMatrixType;

/** Methods to find impostors */
enum ImpostorsMethod
{
	/** Exact method using the whole data */
	ExactSearch,

	/** Approximate method using the last exact impostors set */
	ApproxSearch
};

/**
 * Class ImpostorNode used to represent the sets of impostors. Each of the elements
 * in a set of impostors is an impostor node. An impostor node holds information
 * of the indices of the example, the target and the impostor.
 */
class CImpostorNode
{
	public:

		/**
		 * Standard impostor node construct
		 *
		 * @param ex example index
		 * @param tar target index
		 * @param imp impostor index
		 */
		CImpostorNode(index_t ex, index_t tar, index_t imp);

		/**
		 * The index of the example defines which impostor node is larger
		 * (the larger example index, the larger impostor node). In case
		 * of equality, then the target index decides, and, in the event
		 * of equality in both example and target indices. then the impostors
		 * defines which node is largest.
		 *
		 * @param rhs right hand side argument of the operator
		 * @return whether the lhs argument was smaller than the rhs, equal to,
		 * or larger
		 */
		bool operator<(const CImpostorNode& rhs) const;

	public:

		/** example index */
		index_t example;

		/** target index */
		index_t target;

		/** impostor index */
		index_t impostor;
};

/**
 * Class CLMNNImpl used to hide the implementation details of LMNN.
 */
class CLMNNImpl
{
	public:

		/**
		 * for each feature in x, find its target neighbors; this is, its k
		 * nearest neighbors with the same label as indicated by y
		 */
		static SGMatrix<index_t> find_target_nn(CDenseFeatures<float64_t>* x, CMulticlassLabels* y, int32_t k);

		/** for each pair of features in x, compute their outer product */
		static OuterProductsMatrixType compute_outer_products(CDenseFeatures<float64_t>* x);

		/** sum the outer products indicated by target_nn in the matrix of outer products C */
		static Eigen::MatrixXd sum_outer_products(const OuterProductsMatrixType& C, const SGMatrix<index_t> target_nn);

		/** find the impostors that remain after applying the transformation L */
		static ImpostorsSetType find_impostors(ImpostorsMethod method, CDenseFeatures<float64_t>* x, CMulticlassLabels* y, const Eigen::MatrixXd& L, SGMatrix<index_t> target_nn, const ImpostorsSetType& Nexact);

		/** update the gradient using the last transition in the impostors sets */
		static void update_gradient(Eigen::MatrixXd& G, const OuterProductsMatrixType& C, const ImpostorsSetType& Nc, const ImpostorsSetType& Np, float64_t mu);

		/** take gradient step and project onto positive semi-definite cone if necessary */
		static void gradient_step(Eigen::MatrixXd& L, const Eigen::MatrixXd& G, float64_t stepsize);

		/** compute LMNN objective */
		static float64_t compute_objective(const CDenseFeatures<float64_t>* x, const Eigen::MatrixXd& L, const OuterProductsMatrixType& C, const SGMatrix<index_t> target_nn, const ImpostorsSetType& Nc, float64_t mu);

	private:

		/** find impostors; variant computing the impostors exactly, using all the data */
		static ImpostorsSetType find_impostors_exact(const Eigen::MatrixXd& LX, const Eigen::MatrixXd& sqdists, CMulticlassLabels* y, int32_t n, int32_t k);

		/** find impostors; approximate variant, using the last exact set of impostors */
		static ImpostorsSetType find_impostors_approx(const Eigen::MatrixXd& LX, const Eigen::MatrixXd& sqdists, const ImpostorsSetType& Nexact);

}; /* class CLMNNImpl */

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

#endif /* _LMNNIMPL_H_ */
