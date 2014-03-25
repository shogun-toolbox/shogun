/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDDEGREEPOSITIONSTRINGKERNEL_H___
#define _WEIGHTEDDEGREEPOSITIONSTRINGKERNEL_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/kernel/string/WeightedDegreeStringKernel.h>
#include <shogun/lib/Trie.h>

namespace shogun
{

class CSVM;

/** @brief The Weighted Degree Position String kernel (Weighted Degree kernel
 * with shifts).
 *
 *  The WD-shift kernel of order d compares two sequences \f${\bf x}\f$ and
 *  \f${\bf x'}\f$ of length L by summing all contributions of k-mer matches of
 *  lengths \f$k\in\{1,\dots,d\}\f$, weighted by coefficients \f$\beta_k\f$
 *  allowing for a positional tolerance of up to shift s.
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
class CWeightedDegreePositionStringKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CWeightedDegreePositionStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param max_mismatch maximum mismatch
		 * @param mkl_stepsize MKL stepsize
		 */
		CWeightedDegreePositionStringKernel(
			int32_t size, int32_t degree,
			int32_t max_mismatch=0, int32_t mkl_stepsize=1);

		/** constructor
		 *
		 * @param size cache size
		 * @param weights weights
		 * @param degree degree
		 * @param max_mismatch maximum mismatch
		 * @param shifts position shifts
		 * @param mkl_stepsize MKL stepsize
		 */
		CWeightedDegreePositionStringKernel(
			int32_t size, SGVector<float64_t> weights, int32_t degree,
			int32_t max_mismatch, SGVector<int32_t> shifts,
			int32_t mkl_stepsize=1);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CWeightedDegreePositionStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t degree);

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

		/** return what type of kernel we are
		 *
		 * @return kernel type WEIGHTEDDEGREEPOS
		 */
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREEPOS; }

		/** return the kernel's name
		 *
		 * @return name WeightedDegreePos
		 */
		virtual const char* get_name() const { return "WeightedDegreePositionStringKernel"; }

		/** initialize optimization
		 *
		 * @param p_count count
		 * @param IDX index
		 * @param alphas alphas
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t p_count, int32_t *IDX, float64_t * alphas)
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
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t * alphas, int32_t tree_num,
			int32_t upto_tree=-1);

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
		virtual float64_t compute_optimized(int32_t idx)
		{
			ASSERT(get_is_initialized())
			ASSERT(alphabet)
			ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)
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
		virtual void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0);

		/** clear normal
		 * subkernel functionality
		 */
		virtual void clear_normal()
		{
			if ((opt_type==FASTBUTMEMHUNGRY) && (tries.get_use_compact_terminal_nodes()))
			{
				tries.set_use_compact_terminal_nodes(false) ;
				SG_DEBUG("disabling compact trie nodes with FASTBUTMEMHUNGRY\n")
			}

			if (get_is_initialized())
			{
				if (opt_type==SLOWBUTMEMEFFICIENT)
					tries.delete_trees(true);
				else if (opt_type==FASTBUTMEMHUNGRY)
					tries.delete_trees(false);  // still buggy
				else
					SG_ERROR("unknown optimization type\n")

				set_is_initialized(false);
			}
		}

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(int32_t idx, float64_t weight)
		{
			add_example_to_tree(idx, weight);
			set_is_initialized(true);
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		virtual int32_t get_num_subkernels()
		{
			if (position_weights!=NULL)
				return (int32_t) ceil(1.0*seq_length/mkl_stepsize) ;
			if (length==0)
				return (int32_t) ceil(1.0*get_degree()/mkl_stepsize);
			return (int32_t) ceil(1.0*get_degree()*length/mkl_stepsize) ;
		}

		/** compute by subkernel
		 *
		 * @param idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		inline void compute_by_subkernel(
			int32_t idx, float64_t * subkernel_contrib)
		{
			if (get_is_initialized())
			{
				compute_by_tree(idx, subkernel_contrib);
				return ;
			}

			SG_ERROR("CWeightedDegreePositionStringKernel optimization not initialized\n")
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		inline const float64_t* get_subkernel_weights(int32_t& num_weights)
		{
			num_weights = get_num_subkernels() ;

			SG_FREE(weights_buffer);
			weights_buffer = SG_MALLOC(float64_t, num_weights);

			if (position_weights!=NULL)
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = position_weights[i*mkl_stepsize] ;
			else
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = weights[i*mkl_stepsize] ;

			return weights_buffer ;
		}

		/** set subkernel weights
		 *
		 * @param w weights
		 */
		virtual void set_subkernel_weights(SGVector<float64_t> w)
		{
			float64_t* weights2=w.vector;
			int32_t num_weights2=w.vlen;

			int32_t num_weights = get_num_subkernels() ;
			if (num_weights!=num_weights2)
				SG_ERROR("number of weights do not match\n")

			if (position_weights!=NULL)
				for (int32_t i=0; i<num_weights; i++)
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<seq_length)
							position_weights[i*mkl_stepsize+j] = weights2[i] ;
					}
			else if (length==0)
			{
				for (int32_t i=0; i<num_weights; i++)
					for (int32_t j=0; j<mkl_stepsize; j++)
						if (i*mkl_stepsize+j<get_degree())
							weights[i*mkl_stepsize+j] = weights2[i] ;
			}
			else
			{
				for (int32_t i=0; i<num_weights; i++)
					for (int32_t j=0; j<mkl_stepsize; j++)
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
		float64_t* compute_abs_weights(int32_t & len);

		/** check if tree is initialized
		 *
		 * @return if tree is initialized
		 */
		bool is_tree_initialized() { return tree_initialized; }

		/** get maximum mismatch
		 *
		 * @return maximum mismatch
		 */
		inline int32_t get_max_mismatch() { return max_mismatch; }

		/** get degree
		 *
		 * @return the degree
		 */
		inline int32_t get_degree() { return degree; }

		/** get degree weights
		 *
		 * @param d degree weights will be stored here
		 * @param len number of degree weights will be stored here
		 */
		inline float64_t *get_degree_weights(int32_t& d, int32_t& len)
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
		inline float64_t *get_weights(int32_t& num_weights)
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
		inline float64_t *get_position_weights(int32_t& len)
		{
			len=seq_length;
			return position_weights;
		}

		/** set shifts
		 *
		 * @param shifts new shifts
		 */
		void set_shifts(SGVector<int32_t> shifts);

		/** set weights
		 *
		 * @param new_weights new weights
		 */
		bool set_weights(SGMatrix<float64_t> new_weights);

		/** set wd weights
		 *
		 * @return if setting was successful
		 */
		virtual bool set_wd_weights();

		/** set position weights
		 *
		 * @param pws new position weights
		 * @return if setting was successful
		 */
		virtual void set_position_weights(SGVector<float64_t> pws);

		/** set position weights for left-hand side
		 *
		 * @param pws new position weights
		 * @param len len
		 * @param num num
		 * @return if setting was successful
		 */
		bool set_position_weights_lhs(float64_t* pws, int32_t len, int32_t num);

		/** set position weights for right-hand side
		 *
		 * @param pws new position weights
		 * @param len len
		 * @param num num
		 * @return if setting was successful
		 */
		bool set_position_weights_rhs(float64_t* pws, int32_t len, int32_t num);

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

		/** delete position weights
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights()
		{
			SG_FREE(position_weights);
			position_weights=NULL;
			return true;
		}

		/** delete position weights left-hand side
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights_lhs()
		{
			SG_FREE(position_weights_lhs);
			position_weights_lhs=NULL;
			return true;
		}

		/** delete position weights right-hand side
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights_rhs()
		{
			SG_FREE(position_weights_rhs);
			position_weights_rhs=NULL;
			return true;
		}

		/** compute by tree
		 *
		 * @param idx index
		 * @return computed value
		 */
		virtual float64_t compute_by_tree(int32_t idx);

		/** compute by tree
		 *
		 * @param idx index
		 * @param LevelContrib level contribution
		 */
		virtual void compute_by_tree(int32_t idx, float64_t* LevelContrib);

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
		float64_t* compute_scoring(
			int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
			float64_t* target, int32_t num_suppvec, int32_t* IDX,
			float64_t* weights);

		/** compute consensus string
		 *
		 * @param num_feat number of features
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @return consensus string
		 */
		char* compute_consensus(
			int32_t &num_feat, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas);

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
		float64_t* extract_w(
			int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
			float64_t* w_result, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas);

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
		float64_t* compute_POIM(
			int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
			float64_t* poim_result, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas, float64_t* distrib);

		/** prepare POIM2
		 *
		 * @param distrib distribution
		 */
		void prepare_POIM2(SGMatrix<float64_t> distrib);

		/** compute POIM2
		 *
		 * @param max_degree maximum degree
		 * @param svm SVM
		 */

		void compute_POIM2(int32_t max_degree, CSVM* svm);

		/** get POIM2
		 *
		 * @return poim
		 */
		SGVector<float64_t> get_POIM2();

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
		virtual void add_example_to_tree(
			int32_t idx, float64_t weight);

		/** add example to single tree
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree(
			int32_t idx, float64_t weight, int32_t tree_num);

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

		/** compute with mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_with_mismatch(
			char* avec, int32_t alen, char* bvec, int32_t blen);

		/** compute without mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_without_mismatch(
			char* avec, int32_t alen, char* bvec, int32_t blen);

		/** compute without mismatch matrix
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_without_mismatch_matrix(
			char* avec, int32_t alen, char* bvec, int32_t blen);

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
		float64_t compute_without_mismatch_position_weights(
			char* avec, float64_t *posweights_lhs, int32_t alen,
			char* bvec, float64_t *posweights_rhs, int32_t blen);

		/** remove lhs from kernel */
		virtual void remove_lhs();

		/** Can (optionally) be overridden to post-initialize some
		 *  member variables which are not PARAMETER::ADD'ed.  Make
		 *  sure that at first the overridden method
		 *  BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_post() throw (ShogunException);

	private:
		/** Do basic initialisations like default settings
		 * and registering parameters */
		void init();

	protected:
		/** weights */
		float64_t* weights;
		/** degree */
		int32_t weights_degree;
		/** length */
		int32_t weights_length;

		/** position weights */
		float64_t* position_weights;
		/** position weights len */
		int32_t position_weights_len;

		/** position weights left-hand side */
		float64_t* position_weights_lhs;
		/** position weights len */
		int32_t position_weights_lhs_len;
		/** position weights right-hand side */
		float64_t* position_weights_rhs;
		/** position weights len */
		int32_t position_weights_rhs_len;
		/** position mask */
		bool* position_mask;

		/** weights buffer */
		float64_t* weights_buffer;
		/** MKL stepsize */
		int32_t mkl_stepsize;

		/** degree */
		int32_t degree;
		/** length */
		int32_t length;

		/** maximum mismatch */
		int32_t max_mismatch;
		/** length of sequence */
		int32_t seq_length;

		/** shifts */
		int32_t *shift;
		/** length of shifts */
		int32_t shift_len;
		/** maximum shift */
		int32_t max_shift;

		/** if block computation is used */
		bool block_computation;

		/** (internal) block weights */
		float64_t* block_weights;
		/** WeightedDegree kernel type */
		EWDKernType type;
		/** which degree */
		int32_t which_degree;

		/** tries */
		CTrie<DNATrie> tries;
		/** POIM tries */
		CTrie<POIMTrie> poim_tries;

		/** if tree is initialized */
		bool tree_initialized;
		/** makes add_example_to_tree (ONLY!) use POIMTrie */
		bool use_poim_tries;

		/** temporary memory for the interface to the poim functions */
		float64_t* m_poim_distrib;
		/** temporary memory for the interface to the poim functions */
		float64_t* m_poim;

		/** number of symbols */
		int32_t m_poim_num_sym;
		/** length of string (==num_feat) */
		int32_t m_poim_num_feat;
		/** total size of poim array */
		int32_t m_poim_result_len;

		/** alphabet of features */
		CAlphabet* alphabet;
};
}
#endif /* _WEIGHTEDDEGREEPOSITIONSTRINGKERNEL_H__ */
