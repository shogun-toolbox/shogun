/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Copyright(C) 2014 Thoralf Klein
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/structure/HierarchicalMultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/lib/DynamicArray.h>

using namespace shogun;

CHierarchicalMultilabelModel::CHierarchicalMultilabelModel()
	: CStructuredModel()
{
	init(SGVector<int32_t>(0), false);
}

CHierarchicalMultilabelModel::CHierarchicalMultilabelModel(CFeatures * features,
                CStructuredLabels * labels, SGVector<int32_t> taxonomy,
                bool leaf_nodes_mandatory)
	: CStructuredModel(features, labels)
{
	init(taxonomy, leaf_nodes_mandatory);
}

CStructuredLabels * CHierarchicalMultilabelModel::structured_labels_factory(
        int32_t num_labels)
{
	return new CMultilabelSOLabels(num_labels, m_num_classes);
}

CHierarchicalMultilabelModel::~CHierarchicalMultilabelModel()
{
	SG_FREE(m_children);
}

void CHierarchicalMultilabelModel::init(SGVector<int32_t> taxonomy,
                                        bool leaf_nodes_mandatory)
{
	SG_ADD(&m_num_classes, "num_classes", "Number of (binary) class assignment per label",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_taxonomy, "taxonomy", "Taxonomy of the hierarchy of the labels",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_leaf_nodes_mandatory, "leaf_nodes_mandatory", "Whether internal nodes belong"
	       "to output class or not", MS_NOT_AVAILABLE);
	SG_ADD(&m_root, "root", "Node-id of the ROOT element", MS_NOT_AVAILABLE);

	m_leaf_nodes_mandatory = leaf_nodes_mandatory;
	m_num_classes = 0;

	int32_t num_classes = 0;

	if (m_labels)
	{
		num_classes = ((CMultilabelSOLabels *)m_labels)->get_num_classes();
	}

	REQUIRE(num_classes == taxonomy.vlen, "Number of classes must be equal to taxonomy vector = %d\n",
	        taxonomy.vlen);

	m_taxonomy = SGVector<int32_t>(num_classes);

	m_root = -1;
	int32_t root_node_count = 0;

	for (index_t i = 0; i < num_classes; i++)
	{
		REQUIRE(taxonomy[i] < num_classes && taxonomy[i] >= -1, "parent-id of node-id:%d is taxonomy[%d] = %d,"
		        " but must be within [-1; %d-1] (-1 for root node)\n", i, i,
		        taxonomy[i], num_classes);
		m_taxonomy[i] = taxonomy[i];

		if (m_taxonomy[i] == -1)
		{
			m_root = i;
			root_node_count++;
		}
	}

	if (num_classes)
	{
		REQUIRE(m_root != -1 && root_node_count == 1, "Single ROOT element must be defined "
		        "with parent-id = -1\n");
	}

	// storing all the children of all the nodes in form of array of vectors
	m_children = SG_MALLOC(SGVector<int32_t>, num_classes);

	for (int32_t i = 0; i < num_classes; i++)
	{
		SGVector<int32_t> child_id = m_taxonomy.find(i);
		m_children[i] = child_id;
	}

}

int32_t CHierarchicalMultilabelModel::get_dim() const
{
	int32_t num_classes = ((CMultilabelSOLabels *)m_labels)->get_num_classes();
	int32_t feats_dim = ((CDotFeatures *)m_features)->get_dim_feature_space();

	return num_classes * feats_dim;
}

SGVector<int32_t> CHierarchicalMultilabelModel::get_label_vector(
        SGVector<int32_t> sparse_label)
{
	int32_t num_classes = ((CMultilabelSOLabels *)m_labels)->get_num_classes();

	SGVector<int32_t> label_vector(num_classes);
	label_vector.zero();

	for (index_t i = 0; i < sparse_label.vlen; i++)
	{
		int32_t node_id = sparse_label[i];
		label_vector[node_id] = 1;

		for (int32_t parent_id = m_taxonomy[node_id]; parent_id != -1;
		                parent_id = m_taxonomy[parent_id])
		{
			label_vector[parent_id] = 1;
		}

	}

	return label_vector;
}

SGVector<float64_t> CHierarchicalMultilabelModel::get_joint_feature_vector(
        int32_t feat_idx, CStructuredData * y)
{
	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(y);
	SGVector<int32_t> slabel_data = slabel->get_data();
	SGVector<int32_t> label_vector = get_label_vector(slabel_data);

	SGVector<float64_t> psi(get_dim());
	psi.zero();

	CDotFeatures * dot_feats = (CDotFeatures *)m_features;
	SGVector<float64_t> x = dot_feats->get_computed_dot_feature_vector(feat_idx);
	int32_t feats_dim = dot_feats->get_dim_feature_space();

	for (index_t i = 0; i < label_vector.vlen; i++)
	{
		int32_t label = label_vector[i];

		if (label)
		{
			int32_t offset = i * feats_dim;

			for (index_t j = 0; j < feats_dim; j++)
			{
				psi[offset + j] = x[j];
			}
		}
	}

	return psi;
}

float64_t CHierarchicalMultilabelModel::delta_loss(CStructuredData * y1,
                CStructuredData * y2)
{
	CSparseMultilabel * y1_slabel = CSparseMultilabel::obtain_from_generic(y1);
	CSparseMultilabel * y2_slabel = CSparseMultilabel::obtain_from_generic(y2);

	ASSERT(y1_slabel != NULL);
	ASSERT(y2_slabel != NULL);

	return delta_loss(get_label_vector(y1_slabel->get_data()),
	                  get_label_vector(y2_slabel->get_data()));
}

float64_t CHierarchicalMultilabelModel::delta_loss(SGVector<int32_t> y1,
                SGVector<int32_t> y2)
{
	REQUIRE(y1.vlen == y2.vlen, "Size of both the vectors should be same\n");

	float64_t loss = 0;

	for (index_t i = 0; i < y1.vlen; i++)
	{
		loss += delta_loss(y1[i], y2[i]);
	}

	return loss;
}

float64_t CHierarchicalMultilabelModel::delta_loss(int32_t y1, int32_t y2)
{
	return y1 != y2 ? 1 : 0;
}

void CHierarchicalMultilabelModel::init_primal_opt(
        float64_t regularization,
        SGMatrix<float64_t> &A,
        SGVector<float64_t> a,
        SGMatrix<float64_t> B,
        SGVector<float64_t> &b,
        SGVector<float64_t> lb,
        SGVector<float64_t> ub,
        SGMatrix<float64_t> &C)
{
	C = SGMatrix<float64_t>::create_identity_matrix(get_dim(), regularization);
}

CResultSet * CHierarchicalMultilabelModel::argmax(SGVector<float64_t> w,
                int32_t feat_idx, bool const training)
{
	CDotFeatures * dot_feats = (CDotFeatures *)m_features;
	int32_t feats_dim = dot_feats->get_dim_feature_space();

	CMultilabelSOLabels * multi_labs = (CMultilabelSOLabels *)m_labels;

	if (training)
	{
		m_num_classes = multi_labs->get_num_classes();
	}

	REQUIRE(m_num_classes > 0, "The model needs to be trained before using "
	        "if for prediction\n");

	int32_t dim = get_dim();
	ASSERT(dim == w.vlen);

	// nodes_to_traverse is a dynamic list which keep tracks of which nodes to
	// traverse
	CDynamicArray<int32_t> * nodes_to_traverse = new CDynamicArray<int32_t>();
	SG_REF(nodes_to_traverse);

	// start traversing with the root node
	// insertion of node at the back end
	nodes_to_traverse->push_back(m_root);

	SGVector<int32_t> y_pred_sparse(m_num_classes);
	int32_t count = 0;

	while (nodes_to_traverse->get_num_elements())
	{
		// extraction of the node at the front end
		int32_t node = nodes_to_traverse->get_element(0);
		nodes_to_traverse->delete_element(0);

		float64_t score = dot_feats->dense_dot(feat_idx, w.vector + node * feats_dim,
		                                       feats_dim);

		// if the score is greater than zero, then all the children nodes are
		// to be traversed next
		if (score > 0)
		{
			SGVector<int32_t> child_id = m_children[node];

			// inserting the children nodes at the back end
			for (index_t i = 0; i < child_id.vlen; i++)
			{
				nodes_to_traverse->push_back(child_id[i]);
			}

			if (m_leaf_nodes_mandatory)
			{
				// terminal nodes don't have any children
				if (child_id.vlen == 0)
				{
					y_pred_sparse[count++] = node;
				}
			}
			else
			{
				y_pred_sparse[count++] = node;
			}
		}
	}

	y_pred_sparse.resize_vector(count);

	CResultSet * ret = new CResultSet();
	SG_REF(ret);
	ret->psi_computed = true;

	CSparseMultilabel * y_pred = new CSparseMultilabel(y_pred_sparse);
	SG_REF(y_pred);

	ret->psi_pred = get_joint_feature_vector(feat_idx, y_pred);
	ret->score = SGVector<float64_t>::dot(w.vector, ret->psi_pred.vector, dim);
	ret->argmax = y_pred;

	if (training)
	{
		ret->delta = CStructuredModel::delta_loss(feat_idx, y_pred);
		ret->psi_truth = CStructuredModel::get_joint_feature_vector(feat_idx,
		                 feat_idx);
		ret->score += (ret->delta - SGVector<float64_t>::dot(w.vector,
		                ret->psi_truth.vector, dim));
	}

	SG_UNREF(nodes_to_traverse);

	return ret;
}

