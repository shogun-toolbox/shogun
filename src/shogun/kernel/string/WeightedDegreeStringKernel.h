/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _WEIGHTEDDEGREESTRINGKERNEL_H___
#define _WEIGHTEDDEGREESTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/Trie.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{

/** WD kernel type */
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
};


/** @brief The Weighted Degree String kernel.
 *
 *  The WD kernel of order d compares two sequences \f${\bf x}\f$ and
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
class WeightedDegreeStringKernel: public StringKernel<char>
{
	public:

		/** default constructor
		 *
		 */
		WeightedDegreeStringKernel();


		/** constructor
		 *
		 * @param degree degree
		 * @param type weighted degree kernel type
		 */
		WeightedDegreeStringKernel(int32_t degree, EWDKernType type=E_WD);

		/** constructor
		 *
		 * @param weights kernel's weights
		 */
		WeightedDegreeStringKernel(const SGVector<float64_t>& weights);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		WeightedDegreeStringKernel(
			const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r, int32_t degree);

		~WeightedDegreeStringKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** get WD kernel weighting type
		 *
		 * @return weighting type
		 *
		 *
		 * \sa EWDKernType
		 */
		EWDKernType get_type() const
		{
			return type;
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type WEIGHTEDDEGREE
		 */
		EKernelType get_kernel_type() override { return K_WEIGHTEDDEGREE; }

		/** return the kernel's name
		 *
		 * @return name WeightedDegree
		 */
		const char* get_name() const override {
			return "WeightedDegreeStringKernel";
		}

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param alphas alphas
		 * @return if initializing was successful
		 */
		bool init_optimization(
			int32_t count, int32_t *IDX, float64_t* alphas) override
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
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t* alphas, int32_t tree_num);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		bool delete_optimization() override;

		/** compute optimized
		*
		* @param idx index to compute
		* @return optimized value at given index
		*/
		float64_t compute_optimized(int32_t idx) override
		{
			if (get_is_initialized())
				return compute_by_tree(idx);

			error("CWeightedDegreeStringKernel optimization not initialized");
			return 0;
		}

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
		void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0) override;

		/** clear normal
		 * subkernel functionality
		 */
		void clear_normal() override
		{
			if (get_is_initialized())
			{

				if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
					error("not implemented");

				tries->delete_trees(max_mismatch==0);
				set_is_initialized(false);
			}
		}

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		void add_to_normal(int32_t idx, float64_t weight) override
		{

			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				error("not implemented");

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
		int32_t get_num_subkernels() override
		{
			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				return std::static_pointer_cast<MultitaskKernelMklNormalizer>(normalizer)->get_num_betas();
			if (position_weights.size() > 0)
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
			int32_t idx, float64_t * subkernel_contrib) override
		{

			if (get_is_initialized())
			{

				if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
					error("not implemented");

				compute_by_tree(idx, subkernel_contrib);
				return ;
			}

			error("CWeightedDegreeStringKernel optimization not initialized");
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		inline SGVector<float64_t> get_subkernel_weights() override
		{
			const auto& num_weights = get_num_subkernels();

			weights_buffer = SGVector<float64_t>(num_weights);

			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = std::static_pointer_cast<MultitaskKernelMklNormalizer>(normalizer)->get_beta(i);
			else if (position_weights.size())
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = position_weights[i*mkl_stepsize];
			else
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = weights[i*mkl_stepsize];

			return weights_buffer;
		}

		/** set subkernel weights
		 *
		 * @param w weights
		 */
		void set_subkernel_weights(SGVector<float64_t> w) override
		{
			int32_t num_weights2=w.size();
			int32_t num_weights = get_num_subkernels();
			if (num_weights!=num_weights2)
				error("number of weights do not match");

			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				for (int32_t i=0; i<num_weights; i++)
					std::static_pointer_cast<MultitaskKernelMklNormalizer>(normalizer)->set_beta(i, w[i]);
			else if (position_weights!=NULL)
			{
				for (int32_t i=0; i<num_weights; i++)
				{
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<seq_length)
							position_weights[i*mkl_stepsize+j] = w[i];
					}
				}
			}
			else if (length==0)
			{
				for (int32_t i=0; i<num_weights; i++)
				{
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<get_degree())
							weights[i*mkl_stepsize+j] = w[i];
					}
				}
			}
			else
			{
				for (int32_t i=0; i<num_weights; i++)
				{
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<get_degree()*length)
							weights[i*mkl_stepsize+j] = w[i];
					}
				}
			}
		}

		/** set the current kernel normalizer
		 *
		 * @return if successful
		 */
		bool set_normalizer(std::shared_ptr<KernelNormalizer> normalizer_) override {

			if (normalizer_ && strcmp(normalizer_->get_name(),"MultitaskKernelTreeNormalizer")==0) {
				unset_property(KP_LINADD);
				unset_property(KP_BATCHEVALUATION);
			}
			else
			{
				set_property(KP_LINADD);
				set_property(KP_BATCHEVALUATION);
			}

			return StringKernel<char>::set_normalizer(normalizer_);
		}

		// other kernel tree operations
		/** compute abs weights
		 *
		 * @param len len
		 * @return computed abs weights
		 */
		float64_t *compute_abs_weights(int32_t & len);

		/** compute by tree
		 *
		 * @param idx index
		 * @param LevelContrib level contribution
		 * @return computed value
		 */
		void compute_by_tree(int32_t idx, float64_t *LevelContrib);

		/** check if tree is initialized
		 *
		 * @return if tree is initialized
		 */
		bool is_tree_initialized() { return tree_initialized; }

		/** get degree weights
		 */
		SGMatrix<float64_t> get_degree_weights() const
		{
			return weights;
		}

		/** get weights
		 *
		 * @return weights
		 */
		inline SGMatrix<float64_t> get_weights() const
		{
			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				error("not implemented");

			if (position_weights.size() > 0)
			{
				return SGMatrix<float64_t>(position_weights);
			}
			return weights;
		}

		/** get position weights
		 *
		 * @return position weights
		 */
		inline SGVector<float64_t> get_position_weights() const
		{
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
		 * @param new_weights new weights
		 */
		inline void set_wd_weights(SGVector<float64_t> new_weights)
		{
			SGMatrix<float64_t> matrix = SGMatrix<float64_t>(new_weights.vector,new_weights.vlen,0);
			set_weights(matrix);
			matrix.matrix = NULL;
		}

		/** set weights
		 *
		 * @param new_weights new weights
		 */
		bool set_weights(const SGMatrix<float64_t>& new_weights);

		/** set position weights
		 *
		 * @param pws new position weights
		 * @return if setting was successful
		 */
		bool set_position_weights(const SGVector<float64_t>& pws);

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
			position_weights=SGVector<float64_t>();
			return true;
		}

		/** set maximum mismatch
		 *
		 * @param max new maximum mismatch
		 * @return if setting was successful
		 */
		bool set_max_mismatch(int32_t max);

		/** get maximum mismatch
		 *
		 * @return maximum mismatch
		 */
		inline int32_t get_max_mismatch() const { return max_mismatch; }

		/** set degree
		 *
		 * @param deg new degree
		 * @return if setting was successful
		 */
		inline bool set_degree(int32_t deg) { degree=deg; return true; }

		/** get degree
		 *
		 * @return degree
		 */
		inline int32_t get_degree() const { return degree; }

		/** set if block computation shall be performed
		 *
		 * @param block if block computation shall be performed
		 * @return if setting was successful
		 */
		inline bool set_use_block_computation(bool block)
		{
			block_computation=block;
			return true;
		}

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
		inline bool set_mkl_stepsize(int32_t step)
		{
			if (step<1)
				error("Stepsize must be a positive integer");
			mkl_stepsize=step;
			return true;
		}

		/** get MKL step size
		 *
		 * @return MKL step size
		 */
		inline int32_t get_mkl_stepsize() { return mkl_stepsize; }

		/** set which degree
		 *
		 * @param which which degree
		 * @return if setting was successful
		 */
		inline bool set_which_degree(int32_t which)
		{
			which_degree=which;
			return true;
		}

		/** get which degree
		 *
		 * @return which degree
		 */
		inline int32_t get_which_degree() { return which_degree; }

	protected:
		/** create emtpy tries */
		void create_empty_tries();

		/** add example to tree
		 *
		 * @param idx index
		 * @param weight weight
		 */
		void add_example_to_tree(int32_t idx, float64_t weight);

		/** add example to single tree
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree(
			int32_t idx, float64_t weight, int32_t tree_num);

		/** add example to tree mismatch
		 *
		 * @param idx index
		 * @param weight weight
		 */
		void add_example_to_tree_mismatch(int32_t idx, float64_t weight);

		/** add example to single tree mismatch
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree_mismatch(
			int32_t idx, float64_t weight, int32_t tree_num);

		/** compute by tree
		 *
		 * @param idx index
		 * @return computed value
		 */
		float64_t compute_by_tree(int32_t idx);

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

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

		/** compute using block
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_using_block(char* avec, int32_t alen,
			char* bvec, int32_t blen);

		/** remove lhs from kernel */
		void remove_lhs() override;

	protected:
		/** degree*length weights
		 *length must match seq_length if != 0
		 */
		SGMatrix<float64_t> weights;
		/** degree */
		int32_t weights_degree = 0;
		/** length */
		int32_t weights_length = 0;


		/** position weights */
		SGVector<float64_t> position_weights;
		/** position weights */
		int32_t position_weights_len = 0;
		/** weights buffer */
		SGVector<float64_t> weights_buffer;
		/** MKL step size */
		int32_t mkl_stepsize = 1;
		/** degree */
		int32_t degree = 1;
		/** length */
		int32_t length = 0;

		/** maximum mismatch */
		int32_t max_mismatch = 0;
		/** sequence length */
		int32_t seq_length = 0;

		/** if kernel is initialized */
		bool initialized;

		/** if block computation is used */
		bool block_computation = true;

		/** (internal) block weights */
		SGVector<float64_t> block_weights;
		/** WeightedDegree kernel type */
		EWDKernType type = E_WD;
		/** which degree */
		int32_t which_degree = -1;

		/** tries */
		std::shared_ptr<CTrie<DNATrie>> tries;

		/** if tree is initialized */
		bool tree_initialized = false;
};

}

#endif /* _WEIGHTEDDEGREESTRINGKERNEL_H__ */
