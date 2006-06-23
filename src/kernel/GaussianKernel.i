%module GaussianKernel

%include "features/RealFeatures.i"
%include "kernel/RealKernel.i"

%{
    #include "features/RealFeatures.h" 
	#include "kernel/RealKernel.h"
    #include "kernel/GaussianKernel.h" 
%}

%feature("notabstract") CGaussianKernel;

class CGaussianKernel: public CRealKernel
{
 public:
  CGaussianKernel(LONG size, double width);
  CGaussianKernel(CRealFeatures* l, CRealFeatures* r, LONG size, double width);
  ~CGaussianKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  virtual EKernelType get_kernel_type() { return K_GAUSSIAN; }

  virtual const CHAR* get_name() { return "Gaussian" ; } ;

 protected:
  virtual DREAL compute(INT idx_a, INT idx_b);

 protected:
  double width;
};

