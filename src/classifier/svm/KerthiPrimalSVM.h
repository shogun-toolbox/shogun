#ifndef _KERTHIPRIMALSVM_H___
#define _KERTHIPRIMALSVM_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "lib/Cache.h"

class CKerthiPrimalSVM : public CSVM
{
	public:
		CKerthiPrimalSVM();
		virtual ~CKerthiPrimalSVM();
		virtual bool train();
};

#endif

