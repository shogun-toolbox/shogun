/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDNGRAPH_H_
#define SHOGUNADDNGRAPH_H_

#include <shogun/mathematics/graph/OperatorImplementation.h>
#include <shogun/mathematics/graph/nodes/Add.h>

namespace shogun {
	IGNORE_IN_CLASSLIST class AddNGraph : public AddImpl<AddNGraph, OperatorNGraphBackend>
	{
	public:
		AddNGraph(const std::shared_ptr<Node>& node) : AddImpl(node)
		{
		}

		void build()
		{
			m_ngraph_node = std::make_shared<ngraph::op::Add>();
		}

		void evaluate()
		{
		}
	};
}

#endif