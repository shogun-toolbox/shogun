#ifndef _LDA_H___
#define _LDA_H___

#include "lib/common.h"

#ifdef HAVE_LAPACK
#include "features/Features.h"
#include "classifier/LinearClassifier.h"

class CLDA : public CLinearClassifier
{
	public:
		CLDA();
		virtual ~CLDA();

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

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LDA;
		}
	protected:
		DREAL learn_rate;
		INT max_iter;
};
#endif
#endif
