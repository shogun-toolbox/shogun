#ifndef _GUISVM_H__
#define _GUISVM_H__ 

#include "classifier/svm/SVM.h"

class CGUI ;

class CGUISVM
{

public:
	CGUISVM(CGUI*);
	~CGUISVM();

	bool new_svm(CHAR* param);
	bool train(CHAR* param);
	bool test(CHAR* param);
	bool load(CHAR* param);
	bool save(CHAR* param);
	bool set_C(CHAR* param);
	bool set_weight_epsilon(CHAR* param);

	CLabels* classify(CLabels* output=NULL);
	bool classify_example(INT idx, REAL& result);

	inline CSVM* get_svm() { return svm; }

 protected:
	CGUI* gui;
	CSVM* svm;
	double weight_epsilon;
	double C1;
	double C2;
};
#endif
