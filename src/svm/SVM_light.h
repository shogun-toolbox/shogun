#ifndef _SVMLight_H___
#define _SVMLight_H___

#include "svm/SVM.h"
#include "hmm/HMM.h"
#include "lib/Observation.h"
#include "lib/Observation.h"

class CSVMLight:public CSVM
{
    
public:

    CSVMLight();
    virtual ~CSVMLight();

    virtual bool svm_train(CObservation* train, int kernel_type, double C);
    virtual bool svm_test(CObservation* test, FILE* output);
    virtual bool load_svm(FILE* svm_file, CObservation* test);
    virtual bool save_svm(FILE* svm_file);

};

#endif
