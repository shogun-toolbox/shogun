/*
 * EvaluationResult.h
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#ifndef EVALUATIONRESULT_H_
#define EVALUATIONRESULT_H_

#include <shogun/base/SGObject.h>

namespace shogun {

class CEvaluationResult: public shogun::CSGObject {
public:
	CEvaluationResult();
	virtual ~CEvaluationResult();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const
	{
		return "EvaluationResult";
	}
};

} /* namespace shogun */
#endif /* EVALUATIONRESULT_H_ */
