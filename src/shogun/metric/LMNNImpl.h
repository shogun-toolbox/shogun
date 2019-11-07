/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Heiko Strathmann, Viktor Gal
 */

#ifndef LMNNIMPL_H_
#define LMNNIMPL_H_

#include <shogun/lib/config.h>


#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/distance/EuclideanDistance.h>

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
 * Class LMNNImpl used to hide the implementation details of LMNN.
 */
class LMNNImpl
{
	public:

		/**
		 * check feature and label size, dimensions of the initial transform, etc
		 * if the initial transform has not been initialized, do it using PCA
		 */
		static void check_training_setup(
		    const std::shared_ptr<Features>& features, const std::shared_ptr<Labels>& labels,
		    SGMatrix<float64_t>& init_transform, int32_t k);

		/**
		 * for each feature in x, find its target neighbors; that is, its k
		 * nearest neighbors with the same label as indicated by y
		 */
		static SGMatrix<index_t> find_target_nn(const std::shared_ptr<DenseFeatures<float64_t>>& x, const std::shared_ptr<MulticlassLabels>& y, int32_t k);

		/** sum the outer products indicated by target_nn */
		static SGMatrix<float64_t> sum_outer_products(
		    const std::shared_ptr<DenseFeatures<float64_t>>& x, const SGMatrix<index_t>& target_nn);

		/** find the impostors that remain after applying the transformation L */
		static ImpostorsSetType find_impostors(
		    const std::shared_ptr<DenseFeatures<float64_t>>& x, std::shared_ptr<MulticlassLabels> y,
		    const SGMatrix<float64_t>& L, const SGMatrix<index_t>& target_nn,
		    const int32_t iter, const int32_t correction);

		/** update the gradient using the last transition in the impostors sets */
		static void update_gradient(
		    const std::shared_ptr<DenseFeatures<float64_t>>& x, SGMatrix<float64_t>& G,
		    const ImpostorsSetType& Nc, const ImpostorsSetType& Np,
		    float64_t mu);

		/** take gradient step and project onto positive semi-definite cone if necessary */
		static void gradient_step(
		    SGMatrix<float64_t>& L, const SGMatrix<float64_t>& G,
		    float64_t stepsize, bool diagonal);

		/** correct step size depending on the last fluctuation of the objective */
		static void correct_stepsize(
		    float64_t& stepsize, const SGVector<float64_t> obj,
		    const int32_t iter);

		/**
		 * check if the training should terminate; this can happen due to e.g. convergence reached
		 * (the step size became too small or the objective in the last iterations is roughly constant),
		 * or maximum number of iterations reached
		 */
		static bool check_termination(
		    float64_t stepsize, const SGVector<float64_t> obj, int32_t iter,
		    int32_t maxiter, float64_t stepsize_threshold,
		    float64_t obj_threshold);

	private:

		/** initial default transform given by PCA */
		static SGMatrix<float64_t> compute_pca_transform(const std::shared_ptr<DenseFeatures<float64_t>>& features);

		/**
		 * compute squared distances plus margin between each example and its target neighbors
		 * in the transformed feature space
		 */
		static SGMatrix<float64_t> compute_sqdists(
		    const SGMatrix<float64_t>& L, const SGMatrix<index_t>& target_nn);

		/**
		 * compute squared distances between examples and impostors in the given impostors set
		 * Nexact
		 */
		static SGVector<float64_t> compute_impostors_sqdists(
		    const SGMatrix<float64_t>& L, const ImpostorsSetType& Nexact);

		/** find impostors; variant computing the impostors exactly, using all the data */
		static ImpostorsSetType find_impostors_exact(
		    const SGMatrix<float64_t>& LX, const SGMatrix<float64_t>& sqdists,
		    const std::shared_ptr<MulticlassLabels>& y, const SGMatrix<index_t>& target_nn,
		    int32_t k);

		/** find impostors; approximate variant, using the last exact set of impostors */
		static ImpostorsSetType find_impostors_approx(
		    const SGMatrix<float64_t>& LX, const SGMatrix<float64_t>& sqdists,
		    const ImpostorsSetType& Nexact, const SGMatrix<index_t>& target_nn);

		/** get the indices of the examples whose label is equal to yi */
		static std::vector<index_t> get_examples_label(const std::shared_ptr<MulticlassLabels>& y, float64_t yi);

		/** get the indices of the examples whose label is greater than yi */
		static std::vector<index_t> get_examples_gtlabel(const std::shared_ptr<MulticlassLabels>& y, float64_t yi);

		/**
		 * create Euclidean distance where the lhs features are the features in x indexed
		 * by the elements in a, and the rhs features are the ones indexed by b; caller
		 * is responsible of releasing memory
		 */
		static std::shared_ptr<EuclideanDistance> setup_distance(const std::shared_ptr<DenseFeatures<float64_t>>& x, std::vector<index_t>& a, std::vector<index_t>& b);

		/**
		 * check that k is less than the minimum number of examples in any
		 * class.
		 * k must be less than the number of examples in any class because each
		 * example needs k other examples (the nearest ones) of the same class
		 */
		static void check_maximum_k(std::shared_ptr<Labels> labels, int32_t k);

}; /* class LMNNImpl */

} /* namespace shogun */

#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#endif /* _LMNNIMPL_H_ */
