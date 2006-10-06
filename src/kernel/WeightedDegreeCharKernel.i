%module WeightedDegreeCharKernel

%{
    #include "kernel/WeightedDegreeCharKernel.h" 
%}

%include "kernel/CharKernel.i"
%include "carrays.i"
/* %array_class(double, doubleArray); */
%array_class(DREAL, doubleArray);

%feature("notabstract") CWeightedDegreeCharKernel;

class CWeightedDegreeCharKernel: public CCharKernel {
    public:
        CWeightedDegreeCharKernel(LONG size, DREAL* weights, INT degree, INT max_mismatch, bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1) ;
        CWeightedDegreeCharKernel(CCharFeatures* l, CCharFeatures* r, LONG size, DREAL* weights, INT degree, INT max_mismatch=0, bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1) ;
        ~CWeightedDegreeCharKernel() ;

        virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
        virtual void cleanup();
        bool load_init(FILE* src);
        bool save_init(FILE* dest);

        virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }
        virtual const CHAR* get_name() { return "WeightedDegree" ; } ;
};

%pythoncode 
%{

def createDoubleArray(list):
   array = doubleArray(len(list))
   for i in range(len(list)):
      array[i] = list[i]
   return array

%}



