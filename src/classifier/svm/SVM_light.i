%module SVM_light
%{
 #include "SVM_light.h" 
%}

%include "SVM.i" 
/*
%include "../../kernel/CharKernel.h"
%include "../../kernel/SimpleKernel.h"
%include "../../kernel/Kernel.h"
*/

%include "../../kernel/KernelMachine.i"

%feature("notabstract") CSVMLight;

class CSVMLight:public CSVM {
public:
  CSVMLight();
  virtual ~CSVMLight();
  virtual bool	train();
};

typedef struct model {
    long int    sv_num;	
    long int    at_upper_bound;
    double  b;
    long*	supvec;
    double  *alpha;
    long int    *index;       
    long int    totdoc;       
    CKernel* kernel; 
    double  loo_error,loo_recall,loo_precision; 
    double  xa_error,xa_recall,xa_precision;    
} MODEL;

typedef int FNUM;
typedef double FVAL;  
  
typedef struct learn_parm {
      long   type;                 
     double svm_c;                
	  double eps;                
	  double svm_costratio;        
	  double transduction_posratio;
	  
	  long   biased_hyperplane; 
	  long   sharedslack; 
	  long   svm_maxqpsize;        
	  long   svm_newvarsinqp;
	  long   kernel_cache_size;    
	  double epsilon_crit; 
	  double epsilon_shrink;   
	  long   svm_iter_to_shrink;
	  long   maxiter;           
	  long   remove_inconsistent;
	  long   skip_final_opt_check;
	  long   compute_loo;
	  double rho;       
	  long   xa_depth;   
	  char predfile[200]; 
	  char alphafile[200]; 
	  double epsilon_const;        
	  double epsilon_a;            
	  double opt_precision;
	  long   svm_c_steps;          
	  double svm_c_factor;         
	  double svm_costratio_unlab;
	  double svm_unlabbound;
	  double *svm_cost;            
  } LEARN_PARM;

  typedef struct timing_profile {
	  long int   time_kernel;
	  long int   time_opti;
	  long int   time_shrink;
	  long int   time_update;
	  long int   time_model;
	  long int   time_check;
	  long int   time_select;
  } TIMING;
  

typedef struct shrink_state {                                              
  long   *active;                                      
  long   *inactive_since;                                
  long   deactnum;                          
  double **a_history;      
  long   maxhistory;           
  double *last_a;                   
  double *last_lin;    
} SHRINK_STATE;                                                         

