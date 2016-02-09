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


#ifndef CARTREENODEDATA_H__
#define CARTREENODEDATA_H__

#include <shogun/lib/config.h>

namespace shogun
{
/** @brief structure to store data of a node of
 * CART. This can be used as a template type in
 * TreeMachineNode class. CART algorithm uses nodes
 * of type CTreeMachineNode<CARTreeNodeData>
 */
struct CARTreeNodeData
{
	/** classifying attribute */
	int32_t attribute_id;

	/** feature value(s) required to move into this node */
	SGVector<float64_t> transit_into_values;

	/** classification/regression label of data */
	float64_t node_label;

	/** total weight of training samples passing through this node **/
	float64_t total_weight;

	/** total weight of misclassified samples in node/ weighted sum of squared deviation in case of regression **/
	float64_t weight_minus_node;

	/** total weight of misclassified samples in subtree/ weighted sum of squared deviation in case of regression **/
	float64_t weight_minus_branch;

	/** number of leaves in the subtree beginning at this node **/
	int32_t num_leaves;

	/** constructor */
	CARTreeNodeData()
	{
		attribute_id=-1;
		transit_into_values=SGVector<float64_t>();
		node_label=-1.0;
		total_weight=0.;
		weight_minus_node=0.;
		weight_minus_branch=0.;
		num_leaves=0;
	}

	/** print data
	 * @param data the data to be printed
	 */
	static void print_data(const CARTreeNodeData &data)
	{
		SG_SPRINT("classifying feature index=%d\n", data.attribute_id);
		data.transit_into_values.display_vector(data.transit_into_values.vector,data.transit_into_values.vlen, "transit values");
		SG_SPRINT("total weight=%f\n", data.total_weight);
		SG_SPRINT("errored weight of node=%f\n", data.weight_minus_node);
		SG_SPRINT("errored weight of subtree=%f\n", data.weight_minus_branch);
		SG_SPRINT("number of leaves in subtree=%d\n", data.num_leaves);
	}
};


} /* shogun */

#endif /* CARTREENODEDATA_H__ */
