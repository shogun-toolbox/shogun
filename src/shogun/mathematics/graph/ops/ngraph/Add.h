/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDNGRAPH_H_
#define SHOGUNADDNGRAPH_H_

#include <shogun/mathematics/graph/OperatorImplementation.h>
#include <shogun/mathematics/graph/ops/abstract/AddImpl.h>

#include <ngraph/op/add.hpp>

namespace shogun {
	IGNORE_IN_CLASSLIST class AddNGraph : public AddImpl<AddNGraph, OperatorNGraphBackend>
	{
	public:
		AddNGraph() : AddImpl()
		{
		}

		void evaluate() override
		{
		}
	};
}

#endif
