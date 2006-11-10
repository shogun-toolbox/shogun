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

#ifndef _WEIGHTEDDEGREECHARKERNEL_H___
#define _WEIGHTEDDEGREECHARKERNEL_H___

#include "lib/common.h"
#include "lib/Trie.h"
#include "kernel/SimpleKernel.h"
#include "features/CharFeatures.h"

enum EWDKernType
{
	E_WD=0,
	E_EXTERNAL=1,

	E_BLOCK_CONST=2,
	E_BLOCK_LINEAR=3,
	E_BLOCK_SQPOLY=4,
	E_BLOCK_CUBICPOLY=5,
	E_BLOCK_EXP=6,
	E_BLOCK_LOG=7,
	E_BLOCK_EXTERNAL=8
};

class CWeightedDegreeCharKernel: public CSimpleKernel<CHAR>
{
 public:
  CWeightedDegreeCharKernel(INT size, EWDKernType type, INT degree, INT max_mismatch, 
		  bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1, INT which_deg=-1) ;
  CWeightedDegreeCharKernel(INT size, DREAL* weights, INT degree, INT max_mismatch, 
		  bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1, INT which_deg=-1) ;
  ~CWeightedDegreeCharKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "WeightedDegree" ; } ;

  inline virtual bool init_optimization(INT count, INT *IDX, DREAL* alphas)
  {
	  return init_optimization(count, IDX, alphas, -1);
  }

  /// do initialization for tree_num up to upto_tree, use tree_num=-1 to construct all trees
  virtual bool init_optimization(INT count, INT *IDX, DREAL* alphas, INT tree_num);
  virtual bool delete_optimization() ;
  virtual DREAL compute_optimized(INT idx) 
  { 
    if (get_is_initialized())
      return compute_by_tree(idx); 
    
    CIO::message(M_ERROR, "CWeightedDegreeCharKernel optimization not initialized\n") ;
    return 0 ;
  } ;

  virtual void compute_batch(INT num_vec, INT* vec_idx, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);

  // subkernel functionality
  inline virtual void clear_normal()
  {
	  if (get_is_initialized())
	  {
		  tries.delete_trees(); 
		  set_is_initialized(false);
	  }
  }
  inline virtual void add_to_normal(INT idx, DREAL weight) 
  {
	  if (max_mismatch==0)
		  add_example_to_tree(idx, weight);
	  else
		  add_example_to_tree_mismatch(idx, weight);

	  set_is_initialized(true);
  }
  inline virtual INT get_num_subkernels()
  {
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
	  CIO::message(M_ERROR, "CWeightedDegreeCharKernel optimization not initialized\n") ;
  } ;
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
	  {
		  for (INT i=0; i<num_weights; i++)
		  {
			  for (INT j=0; j<mkl_stepsize; j++)
			  {
				  if (i*mkl_stepsize+j<seq_length)
					  position_weights[i*mkl_stepsize+j] = weights2[i] ;
			  }
		  }
	  }
	  else if (length==0)
	  {
		  for (INT i=0; i<num_weights; i++)
		  {
			  for (INT j=0; j<mkl_stepsize; j++)
			  {
				  if (i*mkl_stepsize+j<get_degree())
					  weights[i*mkl_stepsize+j] = weights2[i] ;
			  }
		  }
	  }
	  else
	  {
		  for (INT i=0; i<num_weights; i++)
		  {
			  for (INT j=0; j<mkl_stepsize; j++)
			  {
				  if (i*mkl_stepsize+j<get_degree()*length)
					  weights[i*mkl_stepsize+j] = weights2[i] ;
			  }
		  }
	  }
  }
  
  // other kernel tree operations  
  DREAL *compute_abs_weights(INT & len);
  void compute_by_tree(INT idx, DREAL *LevelContrib);

  bool is_tree_initialized() { return tree_initialized; }

  inline INT get_max_mismatch() { return max_mismatch; }
  inline INT get_degree() { return degree; }
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
  /// compute positional scoring function, which assigns a weight per position, per symbol in the sequence
  DREAL* compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights);
  //void compute_scoring_helper(struct Trie* tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result);

  bool set_wd_weights_by_type(EWDKernType type);

  void set_wd_weights(DREAL* weights, INT d)
  {
      set_weights(weights,d,0);
  }

  bool set_weights(DREAL* weights, INT d, INT len);
  bool set_position_weights(DREAL* position_weights, INT len=0);

  bool init_block_weights();
  bool init_block_weights_from_wd();
  bool init_block_weights_from_wd_external();
  bool init_block_weights_const();
  bool init_block_weights_linear();
  bool init_block_weights_sqpoly();
  bool init_block_weights_cubicpoly();
  bool init_block_weights_exp();
  bool init_block_weights_log();
  bool init_block_weights_external();

  bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;


 protected:

  void add_example_to_tree(INT idx, DREAL weight);
  void add_example_to_single_tree(INT idx, DREAL weight, INT tree_num);
  void add_example_to_tree_mismatch(INT idx, DREAL weight);
  void add_example_to_single_tree_mismatch(INT idx, DREAL weight, INT tree_num);
  void add_example_to_tree_mismatch_recursion(struct Trie *tree,  DREAL alpha,
											  INT *vec, INT len_rem, 
											  INT depth_rec, INT mismatch_rec) ;
  
  DREAL compute_by_tree(INT idx);

  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/
  DREAL compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  DREAL compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  DREAL compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  DREAL compute_using_block(CHAR* avec, INT alen, CHAR* bvec, INT blen);

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;


 protected:
  ///degree*length weights
  ///length must match seq_length if != 0
  DREAL* weights;
  DREAL* position_weights ;
  DREAL* weights_buffer ;
  INT mkl_stepsize ;
  INT degree;
  INT length;
  
  INT max_mismatch ;
  INT seq_length ;

  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;
  bool block_computation;
  bool use_normalization ;
  
  INT num_block_weights_external;
  DREAL* block_weights_external;

  DREAL* block_weights;
  EWDKernType type;
  INT which_degree;
  
  CTrie tries ;
  bool tree_initialized ;
  
};

#endif
