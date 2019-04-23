/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evangelos Anagnostopoulos, Jacob Walker,
 *          Sergey Lisitsyn, Roman Votyakov, Michele Mazzoni, Heiko Strathmann,
 *          Yuyu Zhang, Evgeniy Andreev, Evan Shelhamer, Wu Lin
 */

#ifndef _COMBINEDKERNEL_H___
#define _COMBINEDKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/List.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>

#include <shogun/features/Features.h>
#include <shogun/features/CombinedFeatures.h>

namespace shogun
{
class Features;
class CombinedFeatures;
class List;
class ListElement;
/**
 * @brief The Combined kernel is used to combine a number of kernels into a
 * single CombinedKernel object by linear combination.
 *
 * It keeps pointers to the added sub-kernels \f$k_m({\bf x}, {\bf x'})\f$ and
 * for each sub-kernel - a kernel specific weight \f$\beta_m\f$.
 *
 * It is especially useful to combine kernels working on different domains and
 * to combine kernels looking at independent features and requires
 * CombinedFeatures to be used.
 *
 * It is defined as:
 *
 * \f[
 *     k_{combined}({\bf x}, {\bf x'}) = \sum_{m=1}^M \beta_m k_m({\bf x}, {\bf x'})
 * \f]
 *
 */
class CombinedKernel : public Kernel
{
	public:
		/** Default constructor */
		CombinedKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param append_subkernel_weights if subkernel weights shall be
		 *        appended
		 */
		CombinedKernel(int32_t size, bool append_subkernel_weights);

		virtual ~CombinedKernel();

		/** initialize kernel. Provided features have to be combined features.
		 * If they are not, all subkernels are tried to be initialised with the
		 * single same passed features objects
		 *
		 * @param lhs features of left-hand side
		 * @param rhs features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type COMBINED
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_COMBINED;
		}

		/** return feature type the kernel can deal with
		 *
		 * @return feature type UNKNOWN
		 */
		virtual EFeatureType get_feature_type()
		{
			return F_UNKNOWN;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class COMBINED
		 */
		virtual EFeatureClass get_feature_class()
		{
			return C_COMBINED;
		}

		/** return the kernel's name
		 *
		 * @return name Combined
		 */
		virtual const char* get_name() const { return "CombinedKernel"; }

		/** list kernels */
		void list_kernels();

		/** get first kernel
		 *
		 * @return first kernel
		 */
		inline std::shared_ptr<Kernel> get_first_kernel()
		{
			return get_kernel(0);
		}

		/** get kernel
		 *
		 * @param idx index of kernel
		 * @return kernel at index idx
		 */
		inline std::shared_ptr<Kernel> get_kernel(int32_t idx)
		{
			if (idx < get_num_kernels())
			{
				return std::static_pointer_cast<Kernel>(kernel_array->get_element(idx));
			}
			else
			{
				return 0;
			}
		}

		/** get last kernel
		 *
		 * @return last kernel
		 */
		inline std::shared_ptr<Kernel> get_last_kernel()
		{
			return get_kernel(get_num_kernels()-1);
		}

		/** insert kernel at position idx
		 *	 idx must be < num_kernels
		 *
		 * @param k kernel
		 * @param idx the index of the position where the kernel should be added
		 * @return if inserting was successful
		 */
		inline bool insert_kernel(std::shared_ptr<Kernel> k, int32_t idx)
		{
			ASSERT(k)
			adjust_num_lhs_rhs_initialized(k);

			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			return kernel_array->insert_element(k, idx);
		}

		/** check if all sub-kernels have given property
		 *
		 * @param p kernel property
		 * @return if kernel has given property
		 */
		virtual bool has_property(EKernelProperty p)
		{
			if (p != KP_LINADD)
				return Kernel::has_property(p);

			if (!kernel_array || !kernel_array->get_num_elements())
				return false;

			bool all_linadd = true;
			for (auto i : range(kernel_array->get_num_elements()))
			{
				auto cur = kernel_array->get_element(i);
				all_linadd &= (std::static_pointer_cast<Kernel>(cur))->has_property(p);

				if (!all_linadd)
					break;
			}

			return all_linadd;
		}

		/** append kernel to the end of the array
		 *
		 * @param k kernel
		 * @return if appending was successful
		 */
		inline bool append_kernel(std::shared_ptr<Kernel> k)
		{
			ASSERT(k)
			adjust_num_lhs_rhs_initialized(k);

			if (!(k->has_property(KP_LINADD)))
				unset_property(KP_LINADD);

			int n = get_num_kernels();
			kernel_array->push_back(k);

			if(enable_subkernel_weight_opt && n+1==get_num_kernels())
				enable_subkernel_weight_learning();

			return n+1==get_num_kernels();
		}


		/** delete kernel
		 *
		 * @param idx the index of the kernel to delete
		 * @return if deleting was successful
		 */
		inline bool delete_kernel(int32_t idx)
		{
			bool succesful_deletion = kernel_array->delete_element(idx);

			if (get_num_kernels()==0)
			{
				num_lhs=0;
				num_rhs=0;
			}

			if(enable_subkernel_weight_opt && succesful_deletion && get_num_kernels()>0)
				enable_subkernel_weight_learning();

			return succesful_deletion;
		}

		/** check if subkernel weights are appended
		 *
		 * @return if subkernel weigths are appended
		 */
		inline bool get_append_subkernel_weights()
		{
			return append_subkernel_weights;
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		inline int32_t get_num_subkernels()
		{
			if (append_subkernel_weights)
			{
				int32_t num_subkernels = 0;

				for (index_t k_idx=0; k_idx<get_num_kernels(); k_idx++)
				{
					auto k = get_kernel(k_idx);
					num_subkernels += k->get_num_subkernels();

				}
				return num_subkernels;
			}
			else
				return get_num_kernels();
		}

		/** get number of contained kernels
		 *
		 * @return number of contained kernels
		 */
		int32_t get_num_kernels()
		{
			return kernel_array->get_num_elements();
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			return initialized;
		}

		/** remove lhs from kernel */
		virtual void remove_lhs();

		/** remove rhs from kernel */
		virtual void remove_rhs();

		/** remove lhs and rhs from kernel */
		virtual void remove_lhs_and_rhs();

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t * weights);

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
		virtual float64_t compute_optimized(int32_t idx);

		/** computes output for a batch of examples in an optimized fashion
		 * (favorable if kernel supports it, i.e. has KP_BATCHEVALUATION.  to
		 * the outputvector target (of length num_vec elements) the output for
		 * the examples enumerated in vec_idx are added. therefore make sure
		 * that it is initialized with ZERO. the following num_suppvec, IDX,
		 * alphas arguments are the number of support vectors, their indices and
		 * weights
		 */
		virtual void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0);

		/** emulates batch computation, via linadd optimization w^t x or even
		 * down to sum_i alpha_i K(x_i,x)
		 *
		 * @param k kernel
		 * @param num_vec number of vectors
		 * @param vec_idx vector index
		 * @param target target
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param weights weights
		 */
		void emulate_compute_batch(
			std::shared_ptr<Kernel> k, int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* weights);

		/** add to normal vector
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(int32_t idx, float64_t weight);

		/** clear normal vector */
		virtual void clear_normal();

		/** compute by subkernel
		 *
		 * @param idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		virtual void compute_by_subkernel(
			int32_t idx, float64_t * subkernel_contrib);

		/** get subkernel weights
		 *
		 * @param num_weights where number of weights is stored
		 * @return subkernel weights
		 */
		virtual const float64_t* get_subkernel_weights(int32_t& num_weights);

		/** get subkernel weights (swig compatible)
		 *
		 * @return subkernel weights
		 */
		virtual SGVector<float64_t> get_subkernel_weights();

		/** set subkernel weights
		 *
		 * @param weights new subkernel weights
		 */
		virtual void set_subkernel_weights(SGVector<float64_t> weights);

		/** set optimization type
		 *
		 * @param t optimization type
		 */
		virtual void set_optimization_type(EOptimizationType t);

		/** precompute all sub-kernels */
		bool precompute_subkernels();

		/** Returns a  casted version of the given kernel. Throws an error
		 * if parameter is not of class CombinedKernel. SG_REF's the returned
		 * kernel
		 *
		 * @param kernel kernel to cast to CombinedKernel
		 * @return casted version of kernel.
		 */
		static std::shared_ptr<CombinedKernel> obtain_from_generic(std::shared_ptr<Kernel> kernel);

		/** return derivative with respect to specified parameter
		 *
		 * @param param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return gradient with respect to parameter
		 */
		SGMatrix<float64_t> get_parameter_gradient(const TParameter* param,
				index_t index=-1);

		/** Get the Kernel array
		 *
		 * @return kernel array
		 */
		inline std::shared_ptr<DynamicObjectArray> get_array()
		{

			return kernel_array;
		}

		/** Returns a list of all the different CombinedKernels produced by the
		* cross-product between the kernel lists The returned list performs
		* reference counting on the contained CombinedKernels.
		*
		* @param kernel_list a list of lists of kernels. Each sub-list must
		* contain kernels of the same type
		*
		* @return a list of CombinedKernels.
		*/
		static std::shared_ptr<List> combine_kernels(std::shared_ptr<List> kernel_list);

		/** Enable to find weight for subkernels during model selection
		 */
		virtual void enable_subkernel_weight_learning();

	protected:
		virtual void init_subkernel_weights();

		/** compute kernel function
		 *
		 * @param x x
		 * @param y y
		 * @return computed kernel function
		 */
		virtual float64_t compute(int32_t x, int32_t y);

		/** adjust the variables num_lhs, num_rhs and initialized
		 * based on the kernel to be appended/inserted
		 *
		 * @param k kernel
		 */
		inline void adjust_num_lhs_rhs_initialized(std::shared_ptr<Kernel> k)
		{
			ASSERT(k)

			if (k->get_num_vec_lhs())
			{
				if (num_lhs)
					ASSERT(num_lhs==k->get_num_vec_lhs())
				num_lhs=k->get_num_vec_lhs();

				if (!get_num_subkernels())
				{
					initialized=true;
#ifdef USE_SVMLIGHT
					cache_reset();
#endif //USE_SVMLIGHT
				}
			}
			else
				initialized=false;

			if (k->get_num_vec_rhs())
			{
				if (num_rhs)
					ASSERT(num_rhs==k->get_num_vec_rhs())
				num_rhs=k->get_num_vec_rhs();

				if (!get_num_subkernels())
				{
					initialized=true;
#ifdef USE_SVMLIGHT
					cache_reset();
#endif //USE_SVMLIGHT
				}
			}
			else
				initialized=false;
		}

	private:
		void init();
		/**
		 * The purpose of this function is to make customkernels aware of any
		 * subsets present, regardless whether the features passed are of type
		 * CombinedFeatures or not
		 * @param lhs combined features
		 * @param rhs rombined features
		 * @param lhs_subset subset present on lhs - pass identity subset if
		 * none
		 * @param rhs_subset subset present on rhs - pass identity subset if
		 * none
		 * @return init succesful
		 */
		bool init_with_extracted_subsets(
		    std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs, SGVector<index_t> lhs_subset,
		    SGVector<index_t> rhs_subset);

	protected:
		/** list of kernels */
		std::shared_ptr<DynamicObjectArray> kernel_array;
		/** support vector count */
		int32_t   sv_count;
		/** support vector index */
		int32_t*  sv_idx;
		/** support vector weights */
		float64_t* sv_weight;
		/** subkernel weights buffers */
		float64_t* subkernel_weights_buffer;
		/** if subkernel weights are appended */
		bool append_subkernel_weights;
		/** whether kernel is ready to be used */
		bool initialized;

		/** weight for subkernels (in log domain) */
		SGVector<float64_t> subkernel_log_weights;

		/** enable to find weight for subkernels during model selection */
		bool enable_subkernel_weight_opt;
		/** update the weight for subkernels */
		bool weight_update;
};
}
#endif /* _COMBINEDKERNEL_H__ */
