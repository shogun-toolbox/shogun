%module SVM_light
%{
 #include "classifier/svm/SVM_light.h" 
%}

%include "lib/common.i"
%include "classifier/svm/SVM.i" 

%feature("notabstract") CSVMLight;

class CSVMLight:public CSVM {
public:
  CSVMLight();
  virtual ~CSVMLight();
  virtual bool train();
};

