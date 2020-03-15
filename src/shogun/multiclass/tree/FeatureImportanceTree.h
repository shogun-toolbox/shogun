/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */

#ifndef _FEATUREIMPORTANCETREE_H__
#define _FEATUREIMPORTANCETREE_H__
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/TreeMachineNode.h>

namespace shogun
{

	/** @brief class FeatureImportanceTree, a mixin class for the tree which
	 * needs computing feature importances. This class is derived from
	 * TreeMachine<NodeType> and stores the feature importances
	 *
	 */
	template <typename NodeType>
	class FeatureImportanceTree : public TreeMachine<NodeType>
	{

	protected:
		void compute_feature_importance(
		    int32_t num_features,
		    const std::shared_ptr<TreeMachineNode<NodeType>>& node)
		{
			m_feature_importances = SGVector<float64_t>(num_features);
			m_feature_importances.zero();
			compute_feature_importance_impl(node);
			float64_t total_num_sample = node->data.total_weight;
			linalg::scale(
			    m_feature_importances, m_feature_importances,
			    1.0 / total_num_sample);
			auto normalizer = linalg::sum(m_feature_importances);
			if (normalizer > 0)
				linalg::scale(
				    m_feature_importances, m_feature_importances,
				    1.0 / normalizer);
		}
		SGVector<float64_t> m_feature_importances;

	public:
		virtual ~FeatureImportanceTree() = default;

	private:
		void compute_feature_importance_impl(
		    const std::shared_ptr<TreeMachineNode<NodeType>>& node)
		{
			const auto& children = node->get_children();
			m_feature_importances[node->data.attribute_id] +=
			    node->data.impurity * node->data.total_weight;
			for (const auto& child : children)
			{
				m_feature_importances[node->data.attribute_id] -=
				    child->data.impurity * child->data.total_weight;
				if (child->data.attribute_id >= 0)
				{
					compute_feature_importance_impl(child);
				}
			}
		}
	};

} // namespace shogun
#endif
