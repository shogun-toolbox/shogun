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
#include <shogun/lib/SGString.h>

namespace shogun
{

/** @brief GradientResult is a container class
 * that returns results from GradientEvaluation.
 * It contains the function value as well as its
 * gradient.
 *  */
class CGradientResult: public CEvaluationResult
{

public:

	/*Constructor*/
	CGradientResult();

	/*Destructor*/
	virtual ~CGradientResult();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const
	{
		return "GradientResult";
	}

	/** return what type of result we are.
	 *
	 *
	 * @return result type
	 */
	virtual EEvaluationResultType get_result_type()
	{
		return GRADIENTEVALUATION_RESULT;
	}

	/*Function value*/
	SGVector<float64_t> quantity;

	/*Function Gradient*/
	CMap<SGString<char>, float64_t> gradient;

	/** Returns the function value
	 * and gradient contained in the object.
	 */
	void print_result()
	{
		SG_SPRINT("Quantity: [");

		for (int i = 0; i < quantity.vlen; i++)
			SG_SPRINT("%f, ", quantity[i]);

		SG_SPRINT("] ");

		SG_SPRINT("Gradient: [");

		for (int i = 0; i < gradient.get_num_elements(); i++)
			SG_SPRINT("%f, ", *(gradient.get_element_ptr(i)));

		SG_SPRINT("]\n");

	}
};

} /* namespace shogun */

#endif /* CGRADIENTRESULT_H_ */
