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

#ifndef _WEIGHTEDDEGREEPOSITIONSTRINGKERNEL_H___
#define _WEIGHTEDDEGREEPOSITIONSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"
#include "kernel/WeightedDegreeStringKernel.h"
#include "lib/Trie.h"

class CSVM ;

/** The WeightedDegreePositionString kernel (Weighted Degree kernel with shifts)
 *  of order d compares two sequences \f${\bf x}\f$ and \f${\bf x'}\f$ of length
 *  L by summing all contributions of k-mer matches of lengths
 *  \f$k\in\{1,\dots,d\}\f$, weighted by coefficients \f$\beta_k\f$ allowing for
 *  a positional tolerance of up to shift s.
 *
 *  It is formally defined as
 * \f{eqnarray*}
 * &&\!\!\!\!\!\!\!k({\bf x}_i,{\bf x}_j)=\sum_{k=1}^d\beta_k\sum_{l=1}^{\!\!\!\!L-k+1\!\!\!\!}\gamma_l\sum_{\begin{array}{c}s=0\\
 *   \!\!\!\!s+l\leq L\!\!\!\!\end{array}}^{S(l)}
 *   \delta_s\;\mu_{k,l,s,{{\bf x}_i},{{\bf x}_j}},\\
 *   &&\!\!\!\!\!\!\!\!\!\! {\footnotesize \mu_{k,l,s,{{\bf x}_i},{{\bf x}_j}}\!\!\! =\!\!
 *   I({\bf u}_{k,l+s}({\bf x}_i)\! =\!{\bf u}_{k,l}({\bf x}_j))\! +\!I({\bf u}_{k,l}({\bf x}_i)\!
 *   =\!{\bf u}_{k,l+s}({\bf x}_j))},\nonumber
 *   \f}
 *   where \f$\beta_j\f$ are the weighting coefficients of the j-mers,
 *   \f$\gamma_l\f$ is a weighting over the
 *   position in the sequence, \f$\delta_s=1/(2(s+1))\f$ is the weight assigned
 *   to shifts (in either direction) of extent s, and S(l) determines
 *   the shift range at position l.
 */
class CWeightedDegreePositionStringKernel: public CStringKernel<CHAR>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param max_mismatch maximum mismatch
		 * @param mkl_stepsize MKL stepsize
		 */
		CWeightedDegreePositionStringKernel(INT size, INT degree,
			INT max_mismatch=0, INT mkl_stepsize=1);

		/** constructor
		 *
		 * @param size cache size
		 * @param weights weights
		 * @param degree degree
		 * @param max_mismatch maximum mismatch
		 * @param shift position shifts
		 * @param shift_len number of shifts
		 * @param mkl_stepsize MKL stepsize
		 */
		CWeightedDegreePositionStringKernel(INT size, DREAL* weights,
			INT degree, INT max_mismatch, INT* shift, INT shift_len,
			INT mkl_stepsize=1);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CWeightedDegreePositionStringKernel(
			CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r,
			INT degree);

		virtual ~CWeightedDegreePositionStringKernel();

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
		 * @return kernel type WEIGHTEDDEGREEPOS
		 */
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREEPOS; }

		/** return the kernel's name
		 *
		 * @return name WeightedDegreePos
		 */
		virtual const CHAR* get_name() { return "WeightedDegreePos" ; } ;

		/** initialize optimization
		 *
		 * @param p_count count
		 * @param IDX index
		 * @param alphas alphas
		 * @return if initializing was successful
		 */
		inline virtual bool init_optimization(INT p_count, INT *IDX, DREAL * alphas)
		{ 
			return init_optimization(p_count, IDX, alphas, -1);
		}

		/** initialize optimization
		 * do initialization for tree_num up to upto_tree, use
		 * tree_num=-1 to construct all trees
		 *
		 * @param count count
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param tree_num which tree
		 * @param upto_tree up to this tree
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(INT count, INT *IDX, DREAL * alphas,
			INT tree_num, INT upto_tree=-1);

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
		inline virtual DREAL compute_optimized(INT idx)
		{ 
			ASSERT(get_is_initialized());
			ASSERT(alphabet);
			ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA);
			return compute_by_tree(idx);
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
				else
					SG_ERROR( "unknown optimization type\n");

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
			add_example_to_tree(idx, weight);
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

			SG_ERROR( "CWeightedDegreePositionStringKernel optimization not initialized\n") ;
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
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

		/** set subkernel weights
		 *
		 * @param weights2 weights
		 * @param num_weights2 number of weights
		 */
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
		/** compute abs weights
		 *
		 * @param len len
		 * @return computed abs weights
		 */
		DREAL *compute_abs_weights(INT & len);

		/** check if tree is initialized
		 *
		 * @return if tree is initialized
		 */
		bool is_tree_initialized() { return tree_initialized; }

		/** get maximum mismatch
		 *
		 * @return maximum mismatch
		 */
		inline INT get_max_mismatch() { return max_mismatch; }

		/** get degree
		 *
		 * @return the degree
		 */
		inline INT get_degree() { return degree; }

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

		/** set shifts
		 *
		 * @param shifts new shifts
		 * @param len number of shifts
		 */
		bool set_shifts(INT* shifts, INT len);

		/** set weights
		 *
		 * @param weights new weights
		 * @param d degree
		 * @param len number of weights
		 */
		virtual bool set_weights(DREAL* weights, INT d, INT len=0);

		/** set wd weights
		 *
		 * @return if setting was successful
		 */
		virtual bool set_wd_weights();

		/** set position weights
		 *
		 * @param position_weights new position weights
		 * @param len number of position weights
		 * @return if setting was successful
		 */
		virtual bool set_position_weights(DREAL* position_weights, INT len=0);

		/** set position weights for left-hand side
		 *
		 * @param pws new position weights
		 * @param len len
		 * @param num num
		 * @return if setting was successful
		 */
		bool set_position_weights_lhs(DREAL* pws, INT len, INT num);

		/** set position weights for right-hand side
		 *
		 * @param pws new position weights
		 * @param len len
		 * @param num num
		 * @return if setting was successful
		 */
		bool set_position_weights_rhs(DREAL* pws, INT len, INT num);

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
		bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;

		/** delete position weights left-hand side
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights_lhs() { delete[] position_weights_lhs ; position_weights_lhs=NULL ; return true ; } ;

		/** delete position weights right-hand side
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights_rhs() { delete[] position_weights_rhs ; position_weights_rhs=NULL ; return true ; } ;

		/** compute by tree
		 *
		 * @param idx index
		 * @return computed value
		 */
		virtual DREAL compute_by_tree(INT idx);

		/** compute by tree
		 *
		 * @param idx index
		 * @param LevelContrib level contribution
		 */
		virtual void compute_by_tree(INT idx, DREAL* LevelContrib);

		/** compute positional scoring function, which assigns a
		 * weight per position, per symbol in the sequence
		 *
		 * @param max_degree maximum degree
		 * @param num_feat number of features
		 * @param num_sym number of symbols
		 * @param target target
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param weights weights
		 * @return computed scores
		 */
		DREAL* compute_scoring(INT max_degree, INT& num_feat, INT& num_sym,
			DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights);

		/** compute consensus string
		 *
		 * @param num_feat number of features
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @return consensus string
		 */
		CHAR* compute_consensus(INT &num_feat, INT num_suppvec, INT* IDX,
			DREAL* alphas);

		/** extract w
		 *
		 * @param max_degree maximum degree
		 * @param num_feat number of features
		 * @param num_sym number of symbols
		 * @param w_result w
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @return w
		 */
		DREAL* extract_w( INT max_degree, INT& num_feat, INT& num_sym,
			DREAL* w_result, INT num_suppvec, INT* IDX, DREAL* alphas);

		/** compute POIM
		 *
		 * @param max_degree maximum degree
		 * @param num_feat number of features
		 * @param num_sym number of symbols
		 * @param poim_result poim
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param distrib distribution
		 * @return computed POIMs
		 */
		DREAL* compute_POIM( INT max_degree, INT& num_feat, INT& num_sym,
			DREAL* poim_result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL* distrib );

		/** prepare POIM2
		 *
		 * @param num_feat number of features
		 * @param num_sym number of symbols
		 * @param distrib distribution
		 */
		void prepare_POIM2(DREAL* distrib, INT num_sym, INT num_feat);		

		/** compute POIM2
		 *
		 * @param max_degree maximum degree
		 * @param svm SVM
		 */

		void compute_POIM2(INT max_degree, CSVM* svm);

		/** get POIM2
		 *
		 * @param poim POIMs (returned)
		 * @param result_len (returned)
		 */
		void get_POIM2(DREAL** poim, INT* result_len);

		/// cleanup POIM2
		void cleanup_POIM2();
		
	protected:
		/** create emtpy tries */
		void create_empty_tries();

		/** add example to tree
		 *
		 * @param idx index
		 * @param weight weight
		 */
		virtual void add_example_to_tree(INT idx, DREAL weight);

		/** add example to single tree
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree(INT idx, DREAL weight, INT tree_num);

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual DREAL compute(INT idx_a, INT idx_b);

		/** compute with mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		DREAL compute_with_mismatch(CHAR* avec, INT alen,
			CHAR* bvec, INT blen);

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

		/** compute without mismatch position weights
		 *
		 * @param avec vector a
		 * @param posweights_lhs position weights left-hand side
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param posweights_rhs position weights right-hand side
		 * @param blen length of vector b
		 * @return computed value
		 */
		DREAL compute_without_mismatch_position_weights(
			CHAR* avec, DREAL *posweights_lhs, INT alen,
			CHAR* bvec, DREAL *posweights_rhs, INT blen);

		/** remove lhs from kernel */
		virtual void remove_lhs();

	protected:
		/** weights */
		DREAL* weights;
		/** position weights */
		DREAL* position_weights;
		/** position weights left-hand side */
		DREAL* position_weights_lhs;
		/** position weights right-hand side */
		DREAL* position_weights_rhs;
		/** position mask */
		bool* position_mask;

		/** weights buffer */
		DREAL* weights_buffer;
		/** MKL stepsize */
		INT mkl_stepsize;

		/** degree */
		INT degree;
		/** length */
		INT length;

		/** maximum mismatch */
		INT max_mismatch;
		/** length of sequence */
		INT seq_length;

		/** shifts */
		INT *shift;
		/** length of shifts */
		INT shift_len;
		/** maximum shift */
		INT max_shift;

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
		CTrie<DNATrie> tries;
		/** POIM tries */
		CTrie<POIMTrie> poim_tries;

		/** if tree is initialized */
		bool tree_initialized;
		/** makes add_example_to_tree (ONLY!) use POIMTrie */
		bool use_poim_tries;

		/** temporary memory for the interface to the poim functions */ 
		DREAL* m_poim_distrib;
		/** temporary memory for the interface to the poim functions */ 
		DREAL* m_poim;

		/** number of symbols */
		INT m_poim_num_sym;
		/** length of string (==num_feat) */
		INT m_poim_num_feat;
		/** total size of poim array */
		INT m_poim_result_len;

		/** alphabet of features */
		CAlphabet* alphabet;
};
#endif /* _WEIGHTEDDEGREEPOSITIONSTRINGKERNEL_H__ */
