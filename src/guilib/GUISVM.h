#ifndef _GUISVM_H__
#define _GUISVM_H__ 

#include "svm/SVM.h"
#include "svm/SVM_light.h"
#include "svm_cplex/SVM_cplex.h"

#ifdef SVMMPI
#include "svm_mpi/mpi_svm.h"
#endif

class CGUI ;

class CGUISVM
{

public:
	CGUISVM(CGUI*);
	~CGUISVM();

	bool new_svm(char* param);
	bool train(char* param);
	bool test(char* param);
	bool load(char* param);
	bool save(char* param);
	bool set_C(char* param);

	inline CSVM* get_svm() { return svm; }

 protected:
	CGUI* gui;
	CSVM* svm;
	double C;
};
#endif
