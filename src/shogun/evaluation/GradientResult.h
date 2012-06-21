/*
 * GradientResult.h
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#ifndef GRADIENTRESULT_H_
#define GRADIENTRESULT_H_

#include "EvaluationResult.h"
#include <shogun/lib/Map.h>
#include <shogun/lib/SGString.h>

namespace shogun {

class CGradientResult: public shogun::CEvaluationResult {
public:
	CGradientResult();
	virtual ~CGradientResult();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const
	{
		return "GradientResult";
	}

	SGVector<float64_t> quantity;
	CMap<SGString<char>, float64_t> gradient;

	void print_result()
	{
		SG_SPRINT("Quantity: [");
		for(int i = 0; i < quantity.vlen; i++)
		{
			SG_SPRINT("%f, ", quantity[i]);
		}

		SG_SPRINT("] ");

		SG_SPRINT("Gradient: [");

		for(int i = 0; i < gradient.get_num_elements(); i++)
		{
			SG_SPRINT("%f, ", *(gradient.get_element_ptr(i)));
		}

		SG_SPRINT("]\n");

	}
};

} /* namespace shogun */
#endif /* GRADIENTRESULT_H_ */
