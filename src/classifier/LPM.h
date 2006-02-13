#ifndef _LPM_H___
#define _LPM_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "classifier/LinearClassifier.h"

class CLPM : public CLinearClassifier
{
	public:
		CLPM();
		virtual ~CLPM();

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

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LPM;
		}
	protected:
		REAL learn_rate;
		INT max_iter;
};
#endif
