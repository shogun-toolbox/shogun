#ifndef _LOCIMPKERNEL_H___
#define _LOCIMPKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

class CLocImpKernel: public CKernel
{
  public:
    CLocImpKernel(long size, int width_, int degree1_, int degree2_) ;
    ~CLocImpKernel() ;

    virtual void init(CFeatures* f);
    virtual void cleanup();
    
 protected:
    /// compute kernel function for features a and b
    /// idx_{a,b} denote the index of the feature vectors
    /// in the corresponding feature object
    virtual REAL compute(CFeatures* a, long idx_a, CFeatures* b, long idx_b);
    /*    compute_kernel*/

    void dot_pyr (double* dpt, const short int* const data1,
		  const short int* const data2, const int num1,
		  const int num2) ;

 protected:
    int width, degree1, degree2 ;

    int PYRAL ;
    double NORM ;
    int* pyra ;
    int* stage1 ;
    int NOF_NTS ;
    int dim ;
};

#define BITS 5

#endif


