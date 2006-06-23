%module SVM
%{
 #include "classifier/svm/LibSVM.h" 
%}

%include "lib/common.i"
%include "classifier/svm/SVM.i" 

%feature("notabstract") CLibSVM;

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
