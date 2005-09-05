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
		virtual ~CPerceptron();

		virtual bool train();

		/// set learn rate of gradient descent training algorithm
		inline void set_learn_rate(REAL r)
		{
			learn_rate=r;
		}

		/// set maximum number of iterations
		inline void set_max_iter(INT i)
		{
			max_iter=i;
		}
	protected:
		REAL learn_rate;
		INT max_iter;
};
#endif
