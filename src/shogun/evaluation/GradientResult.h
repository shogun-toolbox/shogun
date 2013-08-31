/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CGRADIENTRESULT_H_
#define CGRADIENTRESULT_H_

#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/lib/Map.h>

namespace shogun
{

/** @brief Container class that returns results from GradientEvaluation. It
 * contains the function value as well as its gradient.
 */
class CGradientResult : public CEvaluationResult
{
public:
	/** default constructor */
	CGradientResult() : CEvaluationResult() { }

	virtual ~CGradientResult() { }

	/** returns the name of the evaluation result
	 *
	 *  @return name GradientResult
	 */
	virtual const char* get_name() const { return "GradientResult"; }

	/** helper method used to specialize a base class instance
	 *
	 * @param eval_result evaluation result
	 *
	 * @return casted CGradientResult object
	 */
	static CGradientResult* obtain_from_generic(CEvaluationResult* eval_result)
	{
		ASSERT(eval_result);

		REQUIRE(eval_result->get_result_type()==GRADIENTEVALUATION_RESULT,
				"Provided evaluation result is not of type CGradientResult!\n")

		SG_REF(eval_result);
		return (CGradientResult*) eval_result;
	}

	/** return what type of result we are.
	 *
	 * @return result type
	 */
	virtual EEvaluationResultType get_result_type() const
	{
		return GRADIENTEVALUATION_RESULT;
	}

	/** prints the function value and gradient contained in the object */
	virtual void print_result()
	{
		// print quantity
		SG_SPRINT("Quantity: [")

		for (index_t i=0; i<quantity.vlen-1; i++)
			SG_SPRINT("%f, ", quantity[i])

		if (quantity.vlen>0)
			SG_SPRINT("%f", quantity[quantity.vlen-1])

		SG_SPRINT("] ")

		// print gradient wrt parameters
		SG_SPRINT("Gradient: [")

		for (index_t i=0; i<gradient.get_num_elements(); i++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* param_node=
				gradient.get_node_ptr(i);

			// get parameter name
			const char* param_name=param_node->key->m_name;

			// get object name
			const char* object_name=
				parameter_dictionary.get_element(param_node->key)->get_name();

			// get gradient wrt parameter
			SGVector<float64_t> param_gradient=param_node->data;

			SG_PRINT("%s.%s: ", object_name, param_name)

			for (index_t j=0; j<param_gradient.vlen-1; j++)
				SG_SPRINT("%f, ", param_gradient[j])

			if (i==gradient.get_num_elements()-1)
			{
				if (param_gradient.vlen>0)
					SG_PRINT("%f", param_gradient[param_gradient.vlen-1])
			}
			else
			{
				if (param_gradient.vlen>0)
					SG_PRINT("%f; ", param_gradient[param_gradient.vlen-1])
			}
		}

		SG_SPRINT("]\n")
		SG_SPRINT("Total Variables: %i\n", total_variables)
	}

	/** function value */
	SGVector<float64_t> quantity;

	/** function gradient */
	CMap<TParameter*, SGVector<float64_t> > gradient;

	/** which objects do the gradient parameters belong to? */
	CMap<TParameter*, CSGObject*>  parameter_dictionary;

	/** total number of variables represented by the gradient */
	uint32_t total_variables;
};
}
#endif /* CGRADIENTRESULT_H_ */
