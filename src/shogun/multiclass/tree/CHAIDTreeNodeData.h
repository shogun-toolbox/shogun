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
 * of type CTreeMachineNode<CHAIDTreeNodeData>
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

	/** print data
	 * @param data the data to be printed
	 */
	static void print_data(const CHAIDTreeNodeData &data)
	{
		SG_SPRINT("classifying feature index=%d\n", data.attribute_id);
		data.distinct_features.display_vector(data.distinct_features.vector,data.distinct_features.vlen, "distinct feature values");
		data.feature_class.display_vector(data.feature_class.vector,data.feature_class.vlen, "categories of features");
		SG_SPRINT("total weight=%f\n", data.total_weight);
		SG_SPRINT("errored weight of node=%f\n", data.weight_minus_node);
	}
};


} /* shogun */

#endif /* CHAIDTREENODEDATA_H__ */
