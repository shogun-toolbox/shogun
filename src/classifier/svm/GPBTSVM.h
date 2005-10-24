#ifndef _GPBTSVM_H___
#define _GPBTSVM_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

class CGPBTSVM : public CSVM
{
	public:
		CGPBTSVM();
		virtual ~CGPBTSVM();
		virtual bool train();
		inline EClassifierType get_classifier_type() { return CT_GPBT; }

	protected:
		struct svm_model* model;
};
#endif
