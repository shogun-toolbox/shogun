/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _PERCEPTRON_H___
#define _PERCEPTRON_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "classifier/LinearClassifier.h"

class CPerceptron : public CLinearClassifier
{
	public:
		CPerceptron();
		CPerceptron(CRealFeatures* traindat, CLabels* trainlab);
		virtual ~CPerceptron();

		virtual inline EClassifierType get_classifier_type() { return CT_PERCEPTRON; }
		virtual bool train();

		/// set learn rate of gradient descent training algorithm
		inline void set_learn_rate(DREAL r)
		{
			learn_rate=r;
		}

		/// set maximum number of iterations
		inline void set_max_iter(INT i)
		{
			max_iter=i;
		}
	protected:
		DREAL learn_rate;
		INT max_iter;
};
#endif
