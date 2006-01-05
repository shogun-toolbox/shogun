%module CustomKernel%{
 #include "CustomKernel.h" 
%}


%include "Kernel.i"
%feature("notabstract") CCustomKernel;

class CCustomKernel: public CKernel
{
 public:
  CCustomKernel();
  ~CCustomKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Custom,...
  inline virtual EKernelType get_kernel_type() { return K_CUSTOM; }

  /** return feature type the kernel can deal with
  */
  inline virtual EFeatureType get_feature_type()
  {
	  return F_ANY;
  }

  /** return feature class the kernel can deal with
  */
  inline virtual EFeatureClass get_feature_class()
  {
	  return C_ANY;
  }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Custom"; }
};
