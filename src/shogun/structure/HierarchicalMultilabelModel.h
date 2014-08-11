/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Copyright(C) 2014 Thoralf Klein
 * Written(W) 2014 Abinash Panda
 */

#ifndef _HIERARCHICAL_MULTILABEL_MODEL__H__
#define _HIERARCHICAL_MULTILABEL_MODEL__H__

#include <shogun/lib/config.h>
#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/** @brief Class CHierarchicalMultilabelModel represents application specific
 * model and contains application dependent logic for solving hierarchical
 * multilabel classification[1] within a generic SO framework.
 *
 * [1] Lijuan Cai. Multilabel Classification over Category Taxonomies
 *     http://cs.brown.edu/research/pubs/theses/phd/2008/cai.pdf
 *
 * [2] Wei Bi, et.al. Mandatory Leaf Node Prediction in Hierarchical
 *     Multilabel Classification
 *     http://papers.nips.cc/paper/4520-mandatory-leaf-node-prediction-in-hierarchical-multilabel-classification.pdf
 */
class CHierarchicalMultilabelModel : public CStructuredModel
{
public:
	/** default constructor */
	CHierarchicalMultilabelModel();

	/** constructor
	 *
	 * @param features features
	 * @param labels structured labels
	 * @param taxonomy taxonomy is stored in an integer array with:
	 * - length of array = number of nodes
	 * - each element storing only its parent-id taxonomy[node-id] = parent-id
	 * - root node should store its parent-id as -1
	 * @param leaf_nodes_mandatory whether only full paths (from ROOT to
	 * TERMINAL NODES) are only allowed or not
	 * class or not
	 */
	CHierarchicalMultilabelModel(CFeatures * features, CStructuredLabels * labels,
	                             SGVector<int32_t> taxonomy, bool leaf_nodes_mandatory = false);

	/** destructor */
	virtual ~CHierarchicalMultilabelModel();

	/** create empty StructuredLabels object */
	virtual CStructuredLabels * structured_labels_factory(int32_t num_labels = 0);

	/** @return the dimensionality of the joint feature space, i.e., the
	 * dimension of the weight vector \f$w\f$.
	 */
	virtual int32_t get_dim() const;

	/** get joint feature vector
	 *
	 * \f[
	 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y})
	 * \f]
	 *
	 * @param feat_idx index of the feature vector to use
	 * @param y structured labels to use
	 *
	 * @return joint feature vector
	 */
	virtual SGVector<float64_t> get_joint_feature_vector(int32_t feat_idx,
	                CStructuredData * y);

	/** obtain the argmax of
	 *
	 * \f[
	 * \Delta(y_{pred}, y_{truth}) + \langle w, \Psi(x_{truth}, y_{pred} \rangle
	 * \f]
	 *
	 * @param feat_idx index of the feature vector to use
	 * @param y structured label to use
	 */
	virtual CResultSet * argmax(SGVector<float64_t> w, int32_t feat_idx,
	                            bool const training = true);

	/** computes \f$ \Delta(y_{1}, y_{2}) \f$
	 *
	 * @param y1 an instance of structured data
	 * @param y2 another instance of structured data
	 *
	 * @return loss value
	 */
	virtual float64_t delta_loss(CStructuredData * y1, CStructuredData * y2);

	/** initialize the optimization problem
	 *
	 * @param regularization regularization strength
	 * @param A is [-dPsi(y) | -I_N ] with M+N columns => max, M+1 nnz per row
	 * @param a unused input
	 * @param B unused input
	 * @param b upper bound of the constraints, Ax <= b
	 * @param lb lower bounds for w
	 * @param ub upper bounds for w
	 * @param C regularization matrix, w'Cw
	 */
	virtual void init_primal_opt(
	        float64_t regularization,
	        SGMatrix<float64_t> &A,
	        SGVector<float64_t> a,
	        SGMatrix<float64_t> B,
	        SGVector<float64_t> &b,
	        SGVector<float64_t> &lb,
	        SGVector<float64_t> &ub,
	        SGMatrix<float64_t> &C);

	/** @return name of the SGSerializable */
	virtual const char * get_name() const
	{
		return "HierarchicalMultilabelModel";
	}

private:
	int32_t m_num_classes;
	SGVector<int32_t> m_taxonomy;
	bool m_leaf_nodes_mandatory;
	int32_t m_root;
	// array for vectors storing the node-ids of the children
	// m_children[node_id] = vector of node_ids of the children
	//                     = empty vector, if node is a terminal node
	SGVector<int32_t> * m_children;

private:
	void init(SGVector<int32_t> taxonomy, bool leaf_nodes_mandatory);

	/** different versions of delta loss function */
	float64_t delta_loss(SGVector<int32_t> y1, SGVector<int32_t> y2);
	float64_t delta_loss(int32_t y1, int32_t y2);

	/** get the label vector for any label
	 * the label vector would be vector with value 1 for all the nodes
	 * that are ancestor of the given labels else 0
	 *
	 * @return label vector
	 */
	SGVector<int32_t> get_label_vector(SGVector<int32_t> sparse_label);

}; /* class CHierarchicalMultilabelModel */

} /* namespace shogun */

#endif /* _HIERARCHICAL_MULTILABEL_MODEL__H__ */


