/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jiaolong Xu
 */

#ifndef __FACTOR_GRAPH_DATA_GENERATOR_H__
#define __FACTOR_GRAPH_DATA_GENERATOR_H__

#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/io/SGIO.h>

#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/Factor.h>

#include <shogun/structure/MAPInference.h>
#include <shogun/structure/StochasticSOSVM.h>

#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>

#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{
/** @brief Class FactorGraphDataGenerator Create factor graph data for  multiple unit tests
 */
class FactorGraphDataGenerator: public RandomMixin<SGObject>
{
public:
	/** Constructor */
	FactorGraphDataGenerator();

	/** @return class name */
	const char* get_name() const override
	{
		return "FactorGraphDataGenerator";
	}

	/** Define a simple 2 node chain graph */
	std::shared_ptr<FactorGraph> simple_chain_graph();

	/** Convert grid coordinate into 1-d index
	 *
	 * @param x grid coordinate x
	 * @param y grid coordinate y
	 * @param w grid width
	 * @return index in 1-d vector
	 */
	int32_t grid_to_index(int32_t x, int32_t y, int32_t w = 10);

	/** Truncate energy table to ensure submodurality
	 *
	 * @param A value in (0,0)
	 * @param B value in (0,1)
	 * @param C value in (1,0)
	 * @param D value in (1,1)
	 */
	void truncate_energy(float64_t &A, float64_t &B, float64_t &C, float64_t &D);

	/** Define a four nodes chain graph
	 * potentials are randomly generated
	 *
	 * @param assignment_expect expected assignment
	 * @param min_energy_expect expected minimum energies
	 * @param N size of the energy table (e.g., 2x2)
	 */
	std::shared_ptr<FactorGraph> random_chain_graph(SGVector<int> &assignment_expect, float64_t &min_energy_expect, int32_t N = 2);

	/** Define a multiple state tree graph */
	std::shared_ptr<FactorGraph> multi_state_tree_graph();

	/** Generate random data following [1]:
	 * Each example has exactly one label on.
	 * Each label has 40 related binary features.
	 * For an example, if label i is on, 4i randomly chosen features are set to 1
	 *
	 * [1] Finley, Thomas, and Thorsten Joachims.
	 * "Training structural SVMs when exact inference is intractable."
	 * Proceedings of the 25th international conference on Machine learning. ACM, 2008.
	 *
	 * @param len_label label length (10)
	 * @param len_feat feature length (40)
	 * @param size_data training data size (50)
	 * @param feats generated feature matrix
	 * @param labels generated label matrix
	 */
	void generate_data(int32_t len_label, int32_t len_feat, int32_t size_data,
	                   SGMatrix<float64_t> &feats, SGMatrix<int32_t> &labels);

	/** Get fully connected edges
	 *
	 * @param num_classes number of classes
	 * @return matrix of edge
	 */
	SGMatrix< int32_t > get_edges_full(const int32_t num_classes);

	/** Build factor graph
	 *
	 * @param feats features
	 * @param labels labels
	 * @param edge_list edge list
	 * @param v_factor_type factor types
	 * @param fg_feats features for factor graph
	 * @param fg_labels labels for factor graph
	 */
	void build_factor_graph(SGMatrix<float64_t> feats, SGMatrix<int32_t> labels,
	                        SGMatrix< int32_t > edge_list, const std::vector<std::shared_ptr<TableFactorType>> &v_factor_type,
	                        const std::shared_ptr<FactorGraphFeatures>& fg_feats, const std::shared_ptr<FactorGraphLabels>& fg_labels);

	/** Define factor type
	 *
	 * @param num_classes number of classes
	 * @param dim dimension of the feature
	 * @param num_edges number of edegs
	 * @param v_factor_type factor types
	 */
	void define_factor_types(int32_t num_classes, int32_t dim, int32_t num_edges,
	                         std::vector<std::shared_ptr<TableFactorType>> &v_factor_type);

	/** Test sosvm inference algorithm with random data
	 *
	 * @param infer_type type of inference algorithm
	 * @return average training loss (expected to be 0)
	 */
	float64_t test_sosvm(EMAPInferType infer_type);
};
}
#endif
