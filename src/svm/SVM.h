#ifndef _SVM_H___
#define _SVM_H___

#include "hmm/HMM.h"
#include "lib/Observation.h"
#include "lib/Observation.h"
#include "svm/svm_common.h"

class CSVM
{
    
public:

    CSVM();
    ~CSVM();

    bool svm_train(char* svm, CObservation* train, int kernel_type, double C);
    bool svm_test(CObservation* test, FILE* output);
    bool load_svm(FILE* svm_file, CObservation* test);

public:
    bool svm_loaded;
    MODEL svm;
};
#endif
