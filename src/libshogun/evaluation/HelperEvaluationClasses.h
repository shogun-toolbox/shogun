/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef HELPEREVALUATIONCLASSES_H_
#define HELPEREVALUATIONCLASSES_H_

#include "ContingencyTableEvaluation.h"

namespace shogun
{

/** @brief class Accuracy
 * used to measure accuracy of 2-class classifier
 */
class CAccuracy: public CContingencyTableEvaluation
{
	/* constructor */
	CAccuracy() : CContingencyTableEvaluation(ACCURACY) {};
	/* virtual destructor */
	virtual ~CAccuracy() {};
	/* name */
	virtual inline const char* get_name() const { return "Accuracy"; };
};

/** @brief class ErrorRate
 * used to measure error rate of 2-class classifier
 */
class CErrorRate: public CContingencyTableEvaluation
{
	/* constructor */
	CErrorRate() : CContingencyTableEvaluation(ERROR_RATE) {};
	/* virtual destructor */
	virtual ~CErrorRate() {};
	/* name */
	virtual inline const char* get_name() const { return "Error rate"; };
};

/** @brief class BAL
 * used to measure balanced error of 2-class classifier
 */
class CBAL: public CContingencyTableEvaluation
{
	/* constructor */
	CBAL() : CContingencyTableEvaluation(BAL) {};
	/* virtual destructor */
	virtual ~CBAL() {};
	/* name */
	virtual inline const char* get_name() const { return "Balanced error"; };
};

}

#endif /* HELPEREVALUATIONCLASSES_H_ */
