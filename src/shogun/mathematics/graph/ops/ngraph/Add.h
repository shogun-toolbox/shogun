/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDNGRAPH_H_
#define SHOGUNADDNGRAPH_H_

#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/ops/abstract/OperatorImplementation.h>

#include <ngraph/op/add.hpp>

namespace shogun {
	IGNORE_IN_CLASSLIST class AddNGraph : public OperatorImpl<Add>
	{
	public:
		AddNGraph() : OperatorImpl()
		{
		}

		std::string_view get_operator_name() const final
		{
			return "Add";
		}

		void build_implementation() final
		{
		}

		void evaluate() final
		{
		}	
	};
}

#endif
