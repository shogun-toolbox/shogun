/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUTOPERATORIMPL_H_
#define SHOGUNINPUTOPERATORIMPL_H_

#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/OperatorImplementation.h>

namespace shogun {

	template <typename DerivedOperator, typename EngineImplementation>
    IGNORE_IN_CLASSLIST class InputImpl: public OperatorImpl<EngineImplementation, Input>
	{
	public:
		InputImpl(): OperatorImpl<EngineImplementation, Input>() {}

		virtual ~InputImpl() {}

		std::string_view get_operator_name() const override
		{
			return "Input";
		}

		void evaluate() override
		{
			error("Input nodes cannot be run with evaluate. Use evaluate_input(SGContainer) instead");
		}

		void evaluate_input(const std::shared_ptr<Tensor>& tensor)
		{
			static_cast<DerivedOperator*>(this)->evaluate_implementation(tensor);
		}
	};

}

#endif