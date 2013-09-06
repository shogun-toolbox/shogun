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
#ifdef HAVE_LAPACK

#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/distance/EuclideanDistance.h>
#include <Eigen/Dense>

#include <set>
#include <vector>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace shogun
{

struct CImpostorNode;

typedef std::set<CImpostorNode> ImpostorsSetType;

/**
 * Struct ImpostorNode used to represent the sets of impostors. Each of the elements
 * in a set of impostors is an impostor node. An impostor node holds information
 * of the indices of the example, the target and the impostor.
 */
struct CImpostorNode
{
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
		 * check feature and label size, dimensions of the initial transform, etc
		 * if the initial transform has not been initialized, do it using PCA
		 */
		static void check_training_setup(CFeatures* features, const CLabels* labels, SGMatrix<float64_t>& init_transform);

		/**
		 * for each feature in x, find its target neighbors; this is, its k
		 * nearest neighbors with the same label as indicated by y
		 */
		static SGMatrix<index_t> find_target_nn(CDenseFeatures<float64_t>* x, CMulticlassLabels* y, int32_t k);

		/** sum the outer products indicated by target_nn */
		static Eigen::MatrixXd sum_outer_products(CDenseFeatures<float64_t>* x, const SGMatrix<index_t> target_nn);

		/** find the impostors that remain after applying the transformation L */
		static ImpostorsSetType find_impostors(CDenseFeatures<float64_t>* x, CMulticlassLabels* y, const Eigen::MatrixXd& L, const SGMatrix<index_t> target_nn, const uint32_t iter, const uint32_t correction);

		/** update the gradient using the last transition in the impostors sets */
		static void update_gradient(CDenseFeatures<float64_t>* x, Eigen::MatrixXd& G, const ImpostorsSetType& Nc, const ImpostorsSetType& Np, float64_t mu);

		/** take gradient step and project onto positive semi-definite cone if necessary */
		static void gradient_step(Eigen::MatrixXd& L, const Eigen::MatrixXd& G, float64_t stepsize, bool diagonal);

		/** correct step size depending on the last fluctuation of the objective */
		static void correct_stepsize(float64_t& stepsize, const SGVector<float64_t> obj, const uint32_t iter);

	private:

		/** initial default transform given by PCA */
		static SGMatrix<float64_t> compute_pca_transform(CDenseFeatures<float64_t>* features);

		/**
		 * compute squared distances plus margin between each example and its target neighbors
		 * in the transformed feature space
		 */
		static Eigen::MatrixXd compute_sqdists(Eigen::MatrixXd& L, const SGMatrix<index_t> target_nn);

		/**
		 * compute squared distances between examples and impostors in the given impostors set
		 * Nexact
		 */
		static SGVector<float64_t> compute_impostors_sqdists(Eigen::MatrixXd& L, const ImpostorsSetType& Nexact);

		/** find impostors; variant computing the impostors exactly, using all the data */
		static ImpostorsSetType find_impostors_exact(Eigen::MatrixXd& LX, const Eigen::MatrixXd& sqdists, CMulticlassLabels* y, const SGMatrix<index_t> target_nn, int32_t k);

		/** find impostors; approximate variant, using the last exact set of impostors */
		static ImpostorsSetType find_impostors_approx(Eigen::MatrixXd& LX, const Eigen::MatrixXd& sqdists, const ImpostorsSetType& Nexact, const SGMatrix<index_t> target_nn);

		/** get the indices of the examples whose label is equal to yi */
		static std::vector<index_t> get_examples_label(CMulticlassLabels* y, float64_t yi);

		/** get the indices of the examples whose label is greater than yi */
		static std::vector<index_t> get_examples_gtlabel(CMulticlassLabels* y, float64_t yi);

		/**
		 * create Euclidean distance where the lhs features are the features in x indexed
		 * by the elements in a, and the rhs features are the ones indexed by b; caller
		 * is responsible of releasing memory
		 */
		static CEuclideanDistance* setup_distance(CDenseFeatures<float64_t>* x, std::vector<index_t>& a, std::vector<index_t>& b);


}; /* class CLMNNImpl */

} /* namespace shogun */

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#endif /* HAVE_LAPACK */
#endif /* HAVE_EIGEN3 */

#endif /* _LMNNIMPL_H_ */
