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

	protected:
		svm_problem problem;
		svm_parameter param;

		struct svm_model* model;
};
