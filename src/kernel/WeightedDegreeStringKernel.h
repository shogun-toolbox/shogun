/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDDEGREESTRINGKERNEL_H___
#define _WEIGHTEDDEGREESTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Trie.h"
#include "kernel/StringKernel.h"
#include "features/StringFeatures.h"

/** The Weighted Degree kernel of order d compares two sequences \f${\bf x}\f$ and
 *  \f${\bf x'}\f$ of length L by summing all contributions of k-mer matches of
 *  lengths \f$k\in\{1,\dots,d\}\f$, weighted by coefficients \f$\beta_k\f$. It
 *  is defined as
 *  \f[
 *      k({\bf x},{\bf x'})=\sum_{k=1}^d\beta_k\sum_{l=1}^{L-k+1}I({\bf u}_{k,l}({\bf x})={\bf u}_{k,l}({\bf x'})).
 *  \f]
 *      Here, \f${\bf u}_{k,l}({\bf x})\f$ is the string of length k starting at position
 *      l of the sequence \f${\bf x}\f$ and \f$I(\cdot)\f$ is the indicator function
 *      which evaluates to 1 when its argument is true and to 0
 *      otherwise.
 */
class CWeightedDegreeStringKernel: public CStringKernel<CHAR>
{
	public:
		/** constructor
		 *
		 * @param degree degree
		 * @param type weighted degree kernel type
		 */
		CWeightedDegreeStringKernel(INT degree, EWDKernType type=E_WD);

		/** constructor
		 *
		 * @param weights kernel's weights
		 * @param degree degree
		 */
		CWeightedDegreeStringKernel(DREAL* weights, INT degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CWeightedDegreeStringKernel(
			CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT degree);

		virtual ~CWeightedDegreeStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		bool save_init(FILE* dest);

		/** return what type of kernel we are
		 *
		 * @return kernel type WEIGHTEDDEGREE
		 */
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

		/** return the kernel's name
		 *
		 * @return name WeightedDegree
		 */
		virtual const CHAR* get_name() { return "WeightedDegree"; } ;

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param alphas alphas
		 * @return if initializing was successful
		 */
		inline virtual bool init_optimization(INT count, INT *IDX, DREAL* alphas)
		{
			return init_optimization(count, IDX, alphas, -1);
		}

		/** initialize optimization
		 * do initialization for tree_num up to upto_tree, use
		 * tree_num=-1 to construct all trees
		 *
		 * @param count count
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param tree_num which tree
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(INT count, INT *IDX, DREAL* alphas,
			INT tree_num);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization();

		/** compute optimized
	 	*
	 	* @param idx index to compute
	 	* @return optimized value at given index
	 	*/
		virtual DREAL compute_optimized(INT idx)
		{ 
			if (get_is_initialized())
				return compute_by_tree(idx);

			SG_ERROR( "CWeightedDegreeStringKernel optimization not initialized\n");
			return 0;
		}

		/** helper for compute batch
		 *
		 * @param p thread parameter
		 */
		static void* compute_batch_helper(void* p);

		/** compute batch
		 *
		 * @param num_vec number of vectors
		 * @param vec_idx vector index
		 * @param target target
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param factor factor
		 */
		virtual void compute_batch(INT num_vec, INT* vec_idx, DREAL* target,
			INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);

		/** clear normal
		 * subkernel functionality
		 */
		inline virtual void clear_normal()
		{
			if (get_is_initialized())
			{
				tries->delete_trees(max_mismatch==0);
				set_is_initialized(false);
			}
		}

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		inline virtual void add_to_normal(INT idx, DREAL weight)
		{
			if (max_mismatch==0)
				add_example_to_tree(idx, weight);
			else
				add_example_to_tree_mismatch(idx, weight);

			set_is_initialized(true);
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		inline virtual INT get_num_subkernels()
		{
			if (position_weights!=NULL)
				return (INT) ceil(1.0*seq_length/mkl_stepsize) ;
			if (length==0)
				return (INT) ceil(1.0*get_degree()/mkl_stepsize);
			return (INT) ceil(1.0*get_degree()*length/mkl_stepsize) ;
		}

		/** compute by subkernel
		 *
		 * @param idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		inline void compute_by_subkernel(INT idx, DREAL * subkernel_contrib)
		{ 
			if (get_is_initialized())
			{
				compute_by_tree(idx, subkernel_contrib);
				return ;
			}

			SG_ERROR( "CWeightedDegreeStringKernel optimization not initialized\n");
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		inline const DREAL* get_subkernel_weights(INT& num_weights)
		{
			num_weights = get_num_subkernels();

			delete[] weights_buffer ;
			weights_buffer = new DREAL[num_weights];

			if (position_weights!=NULL)
				for (INT i=0; i<num_weights; i++)
					weights_buffer[i] = position_weights[i*mkl_stepsize];
			else
				for (INT i=0; i<num_weights; i++)
					weights_buffer[i] = weights[i*mkl_stepsize];

			return weights_buffer;
		}

		/** set subkernel weights
		 *
		 * @param weights2 weights
		 * @param num_weights2 number of weights
		 */
		inline void set_subkernel_weights(DREAL* weights2, INT num_weights2)
		{
			INT num_weights = get_num_subkernels();
			if (num_weights!=num_weights2)
				SG_ERROR( "number of weights do not match\n");

			if (position_weights!=NULL)
			{
				for (INT i=0; i<num_weights; i++)
				{
					for (INT j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<seq_length)
							position_weights[i*mkl_stepsize+j] = weights2[i];
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
							weights[i*mkl_stepsize+j] = weights2[i];
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
							weights[i*mkl_stepsize+j] = weights2[i];
					}
				}
			}
		}

		// other kernel tree operations
		/** compute abs weights
		 *
		 * @param len len
		 * @return computed abs weights
		 */
		DREAL *compute_abs_weights(INT & len);

		/** compute by tree
		 *
		 * @param idx index
		 * @param LevelContrib level contribution
		 * @return computed value
		 */
		void compute_by_tree(INT idx, DREAL *LevelContrib);

		/** check if tree is initialized
		 *
		 * @return if tree is initialized
		 */
		bool is_tree_initialized() { return tree_initialized; }

		/** get degree weights
		 *
		 * @param d degree weights will be stored here
		 * @param len number of degree weights will be stored here
		 */
		inline DREAL *get_degree_weights(INT& d, INT& len)
		{
			d=degree;
			len=length;
			return weights;
		}

		/** get weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return weights
		 */
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

		/** get position weights
		 *
		 * @param len number of position weights will be stored here
		 * @return position weights
		 */
		inline DREAL *get_position_weights(INT& len)
		{
			len=seq_length;
			return position_weights;
		}

		/** set wd weights
		 *
		 * @param type weighted degree kernel type
		 * @return if setting was successful
		 */
		bool set_wd_weights_by_type(EWDKernType type);

		/** set wd weights
		 *
		 * @param p_weights new eights
		 * @param d degree
		 * @return if setting was successful
		 */
		void set_wd_weights(DREAL* p_weights, INT d)
		{
			set_weights(p_weights,d,0);
		}

		/** set weights
		 *
		 * @param weights new weights
		 * @param d degree
		 * @param len number of weights
		 */
		bool set_weights(DREAL* weights, INT d, INT len);

		/** set position weights
		 *
		 * @param position_weights new position weights
		 * @param len number of position weights
		 * @return if setting was successful
		 */
		bool set_position_weights(DREAL* position_weights, INT len=0);

		/** initialize block weights
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights();

		/** initialize block weights from weighted degree
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_from_wd();

		/** initialize block weights from external weighted degree
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_from_wd_external();

		/** initialize block weights constant
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_const();

		/** initialize block weights linear
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_linear();

		/** initialize block weights squared polynomial
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_sqpoly();

		/** initialize block weights cubic polynomial
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_cubicpoly();

		/** initialize block weights exponential
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_exp();

		/** initialize block weights logarithmic
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_log();

		/** initialize block weights external
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_external();

		/** delete position weights
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights() { delete[] position_weights; position_weights=NULL; return true; }

		/** set maximum mismatch
		 *
		 * @param max new maximum mismatch
		 * @return if setting was succesful
		 */
		bool set_max_mismatch(INT max);

		/** get maximum mismatch
		 *
		 * @return maximum mismatch
		 */
		inline INT get_max_mismatch() { return max_mismatch; }

		/** set degree
		 *
		 * @param deg new degree
		 * @return if setting was successful
		 */
		inline bool set_degree(INT deg) { degree=deg; return true; }

		/** get degree
		 *
		 * @return degree
		 */
		inline INT get_degree() { return degree; }

		/** set if block computation shall be performed
		 *
		 * @param block if block computation shall be performed
		 * @return if setting was successful
		 */
		inline bool set_use_block_computation(bool block) { block_computation=block; return true; }

		/** check if block computation is performed
		 *
		 * @return if block computation is performed
		 */
		inline bool get_use_block_computation() { return block_computation; }

		/** set MKL steps ize
		 *
		 * @param step new step size
		 * @return if setting was successful
		 */
		inline bool set_mkl_stepsize(INT step) { mkl_stepsize=step; return true; }

		/** get MKL step size
		 *
		 * @return MKL step size
		 */
		inline INT get_mkl_stepsize() { return mkl_stepsize; }

		/** set which degree
		 *
		 * @param which which degree
		 * @return if setting was successful
		 */
		inline bool set_which_degree(INT which) { which_degree=which; return true; }

		/** get which degree
		 *
		 * @return which degree
		 */
		inline INT get_which_degree() { return which_degree; }

	protected:
		/** create emtpy tries */
		void create_empty_tries();

		/** add example to tree
		 *
		 * @param idx index
		 * @param weight weight
		 */
		void add_example_to_tree(INT idx, DREAL weight);

		/** add example to single tree
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree(INT idx, DREAL weight, INT tree_num);

		/** add example to tree mismatch
		 *
		 * @param idx index
		 * @param weight weight
		 */
		void add_example_to_tree_mismatch(INT idx, DREAL weight);

		/** add example to single tree mismatch
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree_mismatch(INT idx, DREAL weight, INT tree_num);

		/** add example to tree mismatch recursion
		 *
		 * @param tree tree
		 * @param alpha alpha
		 * @param vec vector
		 * @param len_rem length of rem
		 * @param depth_rec depth rec
		 * @param mismatch_rec mismatch rec
		 */
		void add_example_to_tree_mismatch_recursion(DNATrie *tree,
			DREAL alpha, INT *vec, INT len_rem,
			INT depth_rec, INT mismatch_rec);

		/** compute by tree
		 *
		 * @param idx index
		 * @return computed value
		 */
		DREAL compute_by_tree(INT idx);

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		DREAL compute(INT idx_a, INT idx_b);

		/** compute with mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		DREAL compute_with_mismatch(CHAR* avec, INT alen,
			CHAR* bvec, INT blen) ;

		/** compute without mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		DREAL compute_without_mismatch(CHAR* avec, INT alen,
			CHAR* bvec, INT blen);

		/** compute without mismatch matrix
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		DREAL compute_without_mismatch_matrix(CHAR* avec, INT alen,
			CHAR* bvec, INT blen);

		/** compute using block
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		DREAL compute_using_block(CHAR* avec, INT alen,
			CHAR* bvec, INT blen);

		/** remove lhs from kernel */
		virtual void remove_lhs();

	protected:
		/** degree*length weights
		 *length must match seq_length if != 0
		 */
		DREAL* weights;
		/** position weights */
		DREAL* position_weights;
		/** weights buffer */
		DREAL* weights_buffer;
		/** MKL step size */
		INT mkl_stepsize;
		/** degree */
		INT degree;
		/** length */
		INT length;

		/** maximum mismatch */
		INT max_mismatch;
		/** sequence length */
		INT seq_length;

		/** if kernel is initialized */
		bool initialized;

		/** if block computation is used */
		bool block_computation;

		/** number of external block weights */
		INT num_block_weights_external;
		/** external block weights */
		DREAL* block_weights_external;

		/** (internal) block weights */
		DREAL* block_weights;
		/** WeightedDegree kernel type */
		EWDKernType type;
		/** which degree */
		INT which_degree;

		/** tries */
		CTrie<DNATrie>* tries;

		/** if tree is initialized */
		bool tree_initialized;

		/** alphabet of features */
		CAlphabet* alphabet;
};

#endif /* _WEIGHTEDDEGREESTRINGKERNEL_H__ */
