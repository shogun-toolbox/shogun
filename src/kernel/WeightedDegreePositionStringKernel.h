/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDDEGREEPOSITIONCHARKERNEL_H___
#define _WEIGHTEDDEGREEPOSITIONCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

#include "lib/Trie.h"

class CWeightedDegreePositionStringKernel: public CStringKernel<CHAR>
{
	public:
		CWeightedDegreePositionStringKernel(INT size, DREAL* weights, INT degree, INT max_mismatch, 
				INT * shift, INT shift_len, bool use_norm=true,
				INT mkl_stepsize=1) ;
		//CWeightedDegreePositionStringKernel(INT size, INT degree, INT max_shift, INT max_mismatch=0,
		//		bool use_norm=true, INT mkl_stepsize=1);
		~CWeightedDegreePositionStringKernel() ;

		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();

		/// load and save kernel init_data
		bool load_init(FILE* src);
		bool save_init(FILE* dest);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREEPOS; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "WeightedDegreePos" ; } ;

		inline virtual bool init_optimization(INT p_count, INT *IDX, DREAL * alphas)
		{ 
			return init_optimization(p_count, IDX, alphas, -1);
		}

		/// do initialization for tree_num up to upto_tree, use tree_num=-1 to construct all trees
		virtual bool init_optimization(INT count, INT *IDX, DREAL * alphas, INT tree_num, INT upto_tree=-1);
		virtual bool delete_optimization() ;
		inline virtual DREAL compute_optimized(INT idx) 
		{ 
			ASSERT(get_is_initialized());
			return compute_by_tree(idx); 
		}

		static void* compute_batch_helper(void* p); 
		virtual void compute_batch(INT num_vec, INT* vec_idx, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);

		// subkernel functionality
		inline virtual void clear_normal()
		{
			if ((opt_type==FASTBUTMEMHUNGRY) && (tries.get_use_compact_terminal_nodes())) 
			{
				tries.set_use_compact_terminal_nodes(false) ;
				SG_DEBUG( "disabling compact trie nodes with FASTBUTMEMHUNGRY\n") ;
			}
			if (get_is_initialized())
			{
				if (opt_type==SLOWBUTMEMEFFICIENT)
					tries.delete_trees(true); 
				else if (opt_type==FASTBUTMEMHUNGRY)
					tries.delete_trees(false);  // still buggy
				else {
					SG_ERROR( "unknown optimization type\n");
				}
				set_is_initialized(false);
			}
		}

		inline virtual void add_to_normal(INT idx, DREAL weight) 
		{
			add_example_to_tree(idx, weight);
			set_is_initialized(true);
		}

		inline virtual INT get_num_subkernels()
		{
			//fprintf(stderr, "mkl_stepsize=%i\n", mkl_stepsize) ;
			//exit(-1) ;
			if (position_weights!=NULL)
				return (INT) ceil(1.0*seq_length/mkl_stepsize) ;
			if (length==0)
				return (INT) ceil(1.0*get_degree()/mkl_stepsize);
			return (INT) ceil(1.0*get_degree()*length/mkl_stepsize) ;
		}

		inline void compute_by_subkernel(INT idx, DREAL * subkernel_contrib)
		{ 
			if (get_is_initialized())
			{
				compute_by_tree(idx, subkernel_contrib); 
				return ;
			}

			SG_ERROR( "CWeightedDegreePositionStringKernel optimization not initialized\n") ;
		}

		inline const DREAL* get_subkernel_weights(INT& num_weights)
		{
			num_weights = get_num_subkernels() ;

			delete[] weights_buffer ;
			weights_buffer = new DREAL[num_weights] ;

			if (position_weights!=NULL)
				for (INT i=0; i<num_weights; i++)
					weights_buffer[i] = position_weights[i*mkl_stepsize] ;
			else
				for (INT i=0; i<num_weights; i++)
					weights_buffer[i] = weights[i*mkl_stepsize] ;

			return weights_buffer ;
		}

		inline void set_subkernel_weights(DREAL* weights2, INT num_weights2)
		{
			INT num_weights = get_num_subkernels() ;
			if (num_weights!=num_weights2)
				SG_ERROR( "number of weights do not match\n") ;

			if (position_weights!=NULL)
				for (INT i=0; i<num_weights; i++)
					for (INT j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<seq_length)
							position_weights[i*mkl_stepsize+j] = weights2[i] ;
					}
			else if (length==0)
			{
				for (INT i=0; i<num_weights; i++)
					for (INT j=0; j<mkl_stepsize; j++)
						if (i*mkl_stepsize+j<get_degree())
							weights[i*mkl_stepsize+j] = weights2[i] ;
			}
			else
			{
				for (INT i=0; i<num_weights; i++)
					for (INT j=0; j<mkl_stepsize; j++)
						if (i*mkl_stepsize+j<get_degree()*length)
							weights[i*mkl_stepsize+j] = weights2[i] ;
			}
		}

		// other kernel tree operations  
		DREAL *compute_abs_weights(INT & len);

		bool is_tree_initialized() { return tree_initialized; }

		inline INT get_max_mismatch() { return max_mismatch; }
		inline INT get_degree() { return degree; }


		// weight setting/getting operations
		inline DREAL *get_degree_weights(INT& d, INT& len)
		{
			d=degree;
			len=length;
			return weights;
		}
		inline DREAL *get_weights(INT& num_weights)
		{
			if (position_weights!=NULL)
			{
				num_weights = seq_length ;
				return position_weights ;
			}
			if (length==0)
				num_weights = degree ;
			else
				num_weights = degree*length ;
			return weights;
		}
		inline DREAL *get_position_weights(INT& len)
		{
			len=seq_length;
			return position_weights;
		}
		virtual bool set_weights(DREAL* weights, INT d, INT len=0);
		virtual bool set_position_weights(DREAL* position_weights, INT len=0); 
		bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;

		inline bool get_use_normalization() { return use_normalization; }
		virtual DREAL compute_by_tree(INT idx);
		virtual void compute_by_tree(INT idx, DREAL* LevelContrib);

		/// compute positional scoring function, which assigns a weight per position, per symbol in the sequence
		DREAL* compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights);
		//void compute_scoring_helper(struct Trie* tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result);

	protected:

		virtual void add_example_to_tree(INT idx, DREAL weight);
		void add_example_to_single_tree(INT idx, DREAL weight, INT tree_num);

		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);

		DREAL compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
		DREAL compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
		DREAL compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;

		virtual void remove_lhs() ;
		virtual void remove_rhs() ;

		//DREAL compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos) ;

	protected:
		DREAL* weights;
		DREAL* position_weights ;
		bool* position_mask ;

		INT* counts ;
		DREAL* weights_buffer ;
		INT mkl_stepsize ;

		INT degree;
		INT length;

		INT max_mismatch ;
		INT seq_length ;

		INT *shift ;
		INT shift_len ;
		INT max_shift ;
		DREAL* max_shift_vec ;

		double* sqrtdiag_lhs;
		double* sqrtdiag_rhs;

		bool initialized ;
		bool use_normalization ;

		CTrie tries ;
		bool tree_initialized ;

};

#endif
