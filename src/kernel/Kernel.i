%module Kernel
%{
#include "kernel/Kernel.h" 
%}

%include "carrays.i"
%include "kernel/Kernel.h"
%include "kernel/GaussianKernel.i"


%extend CKernel {

   DREAL* getKernelMatrixReal(){
      int n=0;
      int m=0;
      DREAL *array=NULL; 
      if( self->get_kernel_matrix_real(n,m,array) != NULL) {
         return array;
      }
      return NULL;
   }
}

