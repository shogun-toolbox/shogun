%module Kernel
%{
#include "kernel/Kernel.h" 
%}

%include "lib/common.i"
%include "carrays.i"

class CKernel
{
   public:
      CKernel(KERNELCACHE_IDX size);
      CKernel(CFeatures* lhs, CFeatures* rhs, KERNELCACHE_IDX size);

      virtual ~CKernel();

      DREAL kernel(INT x, INT y);

      virtual DREAL* get_kernel_matrix_real(int &m, int &n, DREAL* target);
      virtual SHORTREAL* get_kernel_matrix_shortreal(int &m, int &n, SHORTREAL* target);

      virtual bool init(CFeatures* lhs, CFeatures* rhs, bool do_init);

      /// clean up your kernel
      virtual void cleanup()=0;

      /// load and save the kernel matrix
      bool load(CHAR* fname);
      bool save(CHAR* fname);

      /// load and save kernel init_data
      virtual bool load_init(FILE* src)=0;
      virtual bool save_init(FILE* dest)=0; 

      /// get left/right hand side of features used in kernel
      inline CFeatures* get_lhs() { return lhs; } ;
      inline CFeatures* get_rhs() { return rhs;  } ;

      inline void cache_reset() { resize_kernel_cache(cache_size) ; } ;

      /// takes all necessary steps if the lhs is removed from kernel
      virtual void remove_lhs();

      /// takes all necessary steps if the rhs is removed from kernel
      virtual void remove_rhs();

      // return what type of kernel we are Linear,Polynomial, Gaussian,...
      virtual EKernelType get_kernel_type()=0 ;

      virtual EFeatureType get_feature_type()=0;

      virtual EFeatureClass get_feature_class()=0;

      // return the name of a kernel
      virtual const CHAR* get_name()=0 ;

      // return the size of the kernel cache
      inline int get_cache_size() { return cache_size; }

      inline int get_max_elems_cache() { return kernel_cache.max_elems; }
      inline int get_activenum_cache() { return kernel_cache.activenum; }

      void list_kernel();

      inline double get_combined_kernel_weight() { return combined_kernel_weight; }
      inline void set_combined_kernel_weight(double nw) { combined_kernel_weight=nw; }

      inline bool get_is_initialized() { return optimization_initialized; }
      inline bool has_property(EKernelProperty p) { return (properties & p) != 0; }

      virtual bool init_optimization(INT count, INT *IDX, DREAL * weights); 
      virtual bool delete_optimization();
      virtual DREAL compute_optimized(INT idx);

      //add vector*factor to 'virtual' normal vector
      virtual void add_to_normal(INT idx, DREAL weight) ;
      virtual void clear_normal();
      virtual INT get_num_subkernels();
      virtual void compute_by_subkernel(INT idx, DREAL * subkernel_contrib);
      virtual const DREAL* get_subkernel_weights(INT& num_weights);
      virtual void set_subkernel_weights(DREAL* weights, INT num_weights);

      virtual bool set_kernel_parameters(INT num, const double* param) { return false; }

      virtual void set_precompute_matrix(bool flag, bool subkernel_flag) { 
         precompute_matrix = flag ; 
         precompute_subkernel_matrix = subkernel_flag ; 

         if (!precompute_matrix)
         {
            delete[] precomputed_matrix ;
            precomputed_matrix = NULL ;
         }
      } ;
      bool get_precompute_matrix() { return precompute_matrix ;  } ;
      bool get_precompute_subkernel_matrix() { return precompute_subkernel_matrix ;  } ;

   protected:
      inline void set_property(EKernelProperty p)
      {
         properties |= p;
      }

      inline void unset_property(EKernelProperty p)
      {
         properties &= (properties | p) ^ p;
      }

      inline void set_is_initialized(bool init) { optimization_initialized=init; }

      /// compute kernel function for features a and b
      /// idx_{a,b} denote the index of the feature vectors
      /// in the corresponding feature object
      virtual DREAL compute(INT x, INT y)=0;

   protected:
      /// feature vectors to occur on left hand side
      CFeatures* lhs;
      /// feature vectors to occur on right hand side
      CFeatures* rhs;

      DREAL combined_kernel_weight ;

      bool optimization_initialized ;

      ULONG  properties;

      // matrix precomputation
      bool precompute_matrix ;
      bool precompute_subkernel_matrix ;
      SHORTREAL * precomputed_matrix ;
      void do_precompute_matrix() ;
};


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

