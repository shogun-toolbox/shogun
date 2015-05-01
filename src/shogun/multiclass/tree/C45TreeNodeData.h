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


#ifndef C45TREENODEDATA_H__
#define C45TREENODEDATA_H__

#include <shogun/lib/config.h>

namespace shogun
{
/** @brief structure to store data of a node of
 * C4.5 tree. This can be used as a template type in
 * TreeMachineNode class. Ex: C4.5 algorithm uses nodes
 * of type CTreeMachineNode<C45TreeNodeData>
 */
struct C45TreeNodeData
{
	/** classifying attribute */
	int32_t attribute_id;

	/** feature value required to move into this node */
	float64_t transit_if_feature_value;

	/** class label of data (-1 for internal nodes) */
	float64_t class_label;

	/** weight of all samples present in the node during training **/
	float64_t total_weight;

	/** weight of all samples present in the node during training not belonging to class_label class **/
	float64_t weight_minus;

	/** constructor */
	C45TreeNodeData()
	{
		attribute_id=-1;
		transit_if_feature_value=-1.0;
		class_label=-1.0;
		total_weight=0.0;
		weight_minus=0.0;
	}

	/** print data
	 * @param data the data to be printed
	 */
	static void print_data(const C45TreeNodeData &data)
	{
		SG_SPRINT("classifying feature index=%d\n", data.attribute_id);
		SG_SPRINT("transit feature value=%f\n", data.transit_if_feature_value);
	}
};


} /* shogun */

#endif /* C45TREENODEDATA_H__ */
