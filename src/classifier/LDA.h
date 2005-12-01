#ifndef _LDA_H___
#define _LDA_H___

#ifdef HAVE_LAPACK
#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "classifier/LinearClassifier.h"

class CLDA : public CLinearClassifier
{
	public:
		CLDA();
		virtual ~CLDA();

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
#endif
