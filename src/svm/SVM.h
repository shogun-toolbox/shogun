#ifndef _SVM_H___
#define _SVM_H___

#include "lib/common.h"
#include "lib/Observation.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 
#include <float.h>
    

class CSVM
{
public:

    CSVM();
    virtual ~CSVM();

    virtual bool svm_train(CObservation* train, int kernel_type, double C)=0;
    virtual bool svm_test(CObservation* test, FILE* output, FILE* rocfile)=0;
    virtual bool load_svm(FILE* svm_file, CObservation* test)=0;
    virtual bool save_svm(FILE* svm_file)=0;

public:
    bool svm_loaded;
};

#endif
