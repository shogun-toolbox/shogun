#ifndef _WDCHARKERNEL_H___
#define _WDCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"

class CWDCharKernel: public CCharKernel
{
	public:
		CWDCharKernel(LONG size, INT degree, INT max_mismatch);
		~CWDCharKernel();

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
		virtual void cleanup();

		/// load and save kernel init_data
		bool load_init(FILE* src);
		bool save_init(FILE* dest);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "WD"; }

		
		virtual bool init_optimization(INT count, INT * IDX, REAL * weights);
		virtual void delete_optimization();
		virtual REAL compute_optimized(INT idx) 
		{ 
			if (get_is_initialized())
				return compute_by_tree(idx); 

			CIO::message(M_ERROR, "CWDCharKernel optimization not initialized\n");
			return 0;
		}

		// other kernel tree operations  
		void prune_tree(struct SuffixTree* p_tree=NULL, int min_usage=2);
		void count_tree_usage(INT idx);
		REAL *compute_abs_weights(INT& len);
		REAL compute_abs_weights_tree(struct SuffixTree* p_tree);

		INT tree_size(struct SuffixTree* p_tree=NULL);
		bool is_tree_initialized() { return tree_initialized; }
		INT get_max_mismatch() { return max_mismatch; }

	protected:
		bool init_matching_weights();
		void add_example_to_tree(INT idx, REAL weight);
		REAL compute_by_tree(INT idx);
		void delete_tree(struct SuffixTree* p_tree=NULL);

		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		REAL compute(INT idx_a, INT idx_b);
		/*    compute_kernel*/

		virtual void remove_lhs();
		virtual void remove_rhs();

	protected:
		REAL* old_weights;
		REAL* matching_weights;
		INT degree;
		INT max_mismatch;
		INT seq_length;

		double* sqrtdiag_lhs;
		double* sqrtdiag_rhs;

		bool initialized;
		bool* match_vector;

		struct SuffixTree** trees;
		bool tree_initialized;
};
#endif
