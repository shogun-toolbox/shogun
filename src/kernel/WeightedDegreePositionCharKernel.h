/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDDEGREEPOSITIONCHARKERNE_H___
#define _WEIGHTEDDEGREEPOSITIONCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"

#include "lib/Trie.h"

class CWeightedDegreePositionCharKernel: public CSimpleKernel<CHAR>
{
public:
	CWeightedDegreePositionCharKernel(LONG size, DREAL* weights, INT degree, INT max_mismatch, 
									  INT * shift, INT shift_len, bool use_norm=false,
									  INT mkl_stepsize=1) ;
	~CWeightedDegreePositionCharKernel() ;
	
	virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
	virtual void cleanup();
	
	/// load and save kernel init_data
	bool load_init(FILE* src);
	bool save_init(FILE* dest);
	
	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREEPOS; }
	
	// return the name of a kernel
	virtual const CHAR* get_name() { return "WeightedDegreePos" ; } ;
	
	inline virtual bool init_optimization(INT count, INT *IDX, DREAL * alphas)
		{ 
			return init_optimization(count, IDX, alphas, -1);
		}
	
	/// do initialization for tree_num up to upto_tree, use tree_num=-1 to construct all trees
	virtual bool init_optimization(INT count, INT *IDX, DREAL * alphas, INT tree_num, INT upto_tree=-1);
	virtual bool delete_optimization() ;
	inline virtual DREAL compute_optimized(INT idx) 
		{ 
			ASSERT(get_is_initialized());
			return compute_by_tree(idx); 
		}
	
	virtual DREAL* compute_batch(INT& num_vec, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);
	
	// subkernel functionality
	inline virtual void clear_normal()
		{
			if ((opt_type==FASTBUTMEMHUNGRY) && (tries.get_use_compact_terminal_nodes())) 
			{
				tries.set_use_compact_terminal_nodes(false) ;
				CIO::message(M_DEBUG, "disabling compact trie nodes with FASTBUTMEMHUNGRY\n") ;
			}
			if (get_is_initialized())
			{
				if (opt_type==SLOWBUTMEMEFFICIENT)
					tries.delete_trees(true); 
				else if (opt_type==FASTBUTMEMHUNGRY)
					tries.delete_trees(false);  // still buggy
				else
					CIO::message(M_ERROR, "unknown optimization type\n");
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
			CIO::message(M_ERROR, "CWeightedDegreePositionCharKernel optimization not initialized\n") ;
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
				CIO::message(M_ERROR, "number of weights do not match\n") ;
			
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
	/// absolute sum of weights, depth level has to be specified
	DREAL compute_abs_weights_tree(struct Trie * p_tree, INT depth);
	
	bool is_tree_initialized() { return tree_initialized; }
	
	inline INT get_max_mismatch() { return max_mismatch; }
	inline INT get_degree() { return degree; }
	
	Trie* get_tree_at_position(INT i);
	void count( const DREAL w, const INT depth, const struct TreeParseInfo info, const INT p, INT* x, const INT k );
	void traverse( const struct Trie* tree, const INT p, struct TreeParseInfo info, const INT depth, INT* const x, const INT k );
	//void count( const DREAL w, const INT p, const INT depth, INT* x, const INT k, DREAL* C_k, DREAL* L_k, DREAL* R_k );
	//void traverse( const struct Trie* tree, INT p, const INT depth, INT* x, const INT k, DREAL* C_k, DREAL* L_k, DREAL* R_k );
	
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
	bool set_weights(DREAL* weights, INT d, INT len=0);
	bool set_position_weights(DREAL* position_weights, INT len=0); 
	bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;
	
	DREAL compute_by_tree(INT idx);
	void compute_by_tree(INT idx, DREAL* LevelContrib);
	
	/// compute positional scoring function, which assigns a weight per position, per symbol in the sequence
	DREAL* compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights);
	//void compute_scoring_helper(struct Trie* tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result);
	
protected:
	
	virtual void add_example_to_tree(INT idx, DREAL weight);
	void add_example_to_single_tree(INT idx, DREAL weight, INT tree_num);
	
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	DREAL compute(INT idx_a, INT idx_b);
	
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
	
	double* sqrtdiag_lhs;
	double* sqrtdiag_rhs;
	
	bool initialized ;
	bool use_normalization ;
	
	CTrie tries ;
	bool tree_initialized ;
	
};

/* computes the simple kernel between position seq_pos and tree tree_pos */
/*
inline DREAL CWeightedDegreePositionCharKernel::compute_by_tree_helper(INT* vec, INT len, INT seq_pos, 
									   INT tree_pos,
									   INT weight_pos)
{
  DREAL sum=0 ;
  
  if ((position_weights!=NULL) && (position_weights[weight_pos]==0))
    return sum;
  
  struct Trie *tree = trees[tree_pos] ;
  ASSERT(tree!=NULL) ;
  
  if (length==0) // weights is a vector (1 x degree)
    {
      for (INT j=0; seq_pos+j < len; j++)
	{
	  if ((j<degree-1) && (tree->children[vec[seq_pos+j]]!=NO_CHILD))
	    {
	      tree=tree->children[vec[seq_pos+j]];
	      sum += tree->weight * weights[j] ;
	    }
	  else
	    {
	      if (j==degree-1)
		sum += tree->child_weights[vec[seq_pos+j]] * weights[j] ;
	      break;
	    }
	} 
    }
  else // weights is a matrix (len x degree)
    {
      if (!position_mask)
	{		
	  position_mask = new bool[len] ;
	  for (INT i=0; i<len; i++)
	    {
	      position_mask[i]=false ;
	      
	      for (INT j=0; j<degree; j++)
		if (weights[i*degree+j]!=0.0)
		  {
		    position_mask[i]=true ;
		    break ;
		  }
	    }
	}
      if (position_mask[weight_pos]==0)
	return 0 ;
      
      for (INT j=0; seq_pos+j<len; j++)
	{
	  if ((j<degree-1) && (tree->children[vec[seq_pos+j]]!=NO_CHILD))
	    {
	      tree=&TreeMem[tree->children[vec[seq_pos+j]]];
	      sum += tree->weight * weights[j+weight_pos*degree] ;
	    }
	  else
	    {
	      if (j==degree-1)
		sum += tree->child_weights[vec[seq_pos+j]] * weights[j+weight_pos*degree] ;
	      break ;
	    }
	} 
    }
  
  if (position_weights!=NULL)
    return sum*position_weights[weight_pos] ;
  else
    return sum ;
}
*/

#endif
