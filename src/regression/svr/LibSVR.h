#ifndef _LIBSVR_H___
#define _LIBSVR_H___
#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

class CLibSVR : public CSVM
{
	public:
		CLibSVR();
		virtual ~CLibSVR();
		virtual bool train();
		inline EClassifierType get_classifier_type() { return CT_NONE; }

	protected:
		svm_problem problem;
		svm_parameter param;

		struct svm_model* model;
};
#endif

