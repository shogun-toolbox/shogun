/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDOPERATORIMPL_H_
#define SHOGUNADDOPERATORIMPL_H_

#include <shogun/mathematics/graph/OperatorImplementation.h>

namespace shogun {
	template <typename DerivedOperator, typename EngineImplementation>
	IGNORE_IN_CLASSLIST class AddImpl: public OperatorImpl<EngineImplementation>
	{
	public:
		AddImpl(): OperatorImpl<EngineImplementation>() {}

		virtual ~AddImpl() {}
		
		std::string_view get_operator_name() const override
		{
			return "Add";
		}

	};
}

#endif