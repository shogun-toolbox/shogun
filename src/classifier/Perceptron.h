/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _PERCEPTRON_H___
#define _PERCEPTRON_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/DotFeatures.h"
#include "classifier/LinearClassifier.h"

/** class Perceptron */
class CPerceptron : public CLinearClassifier
{
	public:
		/** default constructor */
		CPerceptron();

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CPerceptron(CDotFeatures* traindat, CLabels* trainlab);
		virtual ~CPerceptron();

		/** get classifier type
		 *
		 * @return classifier type PERCEPTRON
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_PERCEPTRON; }

		/** train classifier
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/// set learn rate of gradient descent training algorithm
		inline void set_learn_rate(float64_t r)
		{
			learn_rate=r;
		}

		/// set maximum number of iterations
		inline void set_max_iter(int32_t i)
		{
			max_iter=i;
		}

	protected:
		/** learning rate */
		float64_t learn_rate;
		/** maximum number of iterations */
		int32_t max_iter;
};
#endif
