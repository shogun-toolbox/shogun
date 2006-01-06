%module SVM_light
%{
 #include "classifier/svm/SVM_light.h" 
%}

%include "classifier/svm/SVM.i" 
/*
%include "kernel/CharKernel.h"
%include "kernel/SimpleKernel.h"
%include "kernel/Kernel.h"
*/

%include "kernel/KernelMachine.i"

%feature("notabstract") CSVMLight;

class CSVMLight:public CSVM {
public:
  CSVMLight();
  virtual ~CSVMLight();
  virtual bool	train();
};


typedef int FNUM;
typedef double FVAL;  
