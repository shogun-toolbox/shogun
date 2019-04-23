/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */


#ifndef CHAIDTREENODEDATA_H__
#define CHAIDTREENODEDATA_H__

#include <shogun/lib/config.h>

namespace shogun
{
/** @brief structure to store data of a node of
 * CHAID. This can be used as a template type in
 * TreeMachineNode class. CHAID algorithm uses nodes
 * of type TreeMachineNode<CHAIDTreeNodeData>
 */
struct CHAIDTreeNodeData
{
	/** classifying attribute */
	int32_t attribute_id;

	/** distinct feature values possible for attribute_id */
	SGVector<float64_t> distinct_features;

	/** class to which each distinct feature type is assigned */
	SGVector<int32_t> feature_class;

	/** label representative of data in node */
	float64_t node_label;

	/** total weight of training samples passing through this node **/
	float64_t total_weight;

	/** total weight of misclassified samples in node **/
	float64_t weight_minus_node;

	/** constructor */
	CHAIDTreeNodeData()
	{
		attribute_id=-1;
		distinct_features=SGVector<float64_t>();
		feature_class=SGVector<int32_t>();
		node_label=-1.0;
		total_weight=0.;
		weight_minus_node=0.;
	}
};

template<class T>
constexpr void register_params(CHAIDTreeNodeData& n, T* o)
{
	o->watch_param("attribute_id", &n.attribute_id, AnyParameterProperties("classifying feature index"));
	o->watch_param("distinct_features", &n.distinct_features, AnyParameterProperties("distinct feature values possible for attribute_id"));
	o->watch_param("feature_class", &n.feature_class, AnyParameterProperties("class to which each distinct feature type is assigned"));
	o->watch_param("node_label", &n.node_label, AnyParameterProperties("node label"));
	o->watch_param("total_weight", &n.total_weight, AnyParameterProperties("total weight"));
	o->watch_param("weight_minus_node", &n.weight_minus_node,
		AnyParameterProperties("total weight of misclassified samples in node"));
}

} /* shogun */

#endif /* CHAIDTREENODEDATA_H__ */
