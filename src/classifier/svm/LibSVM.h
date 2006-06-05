#ifndef _LIBSVM_H___
#define _LIBSVM_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

class CLibSVM : public CSVM
{
	public:
		CLibSVM();
		virtual ~CLibSVM();
		virtual bool train();
		inline EClassifierType get_classifier_type() { return CT_LIBSVM; }

	protected:
		svm_problem problem;
		svm_parameter param;

		struct svm_model* model;
};
#endif
