#ifndef _CPLEXSVM_H___
#define _CPLEXSVM_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/Cache.h"

class CCPLEXSVM : public CSVM
{
	public:
		CCPLEXSVM();
		virtual ~CCPLEXSVM();
		virtual bool train();

		inline EClassifierType get_classifier_type() { return CT_CPLEXSVM; }
};

#endif
