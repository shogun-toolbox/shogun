/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

#ifndef _DIRECTORKERNEL_H___
#define _DIRECTORKERNEL_H___

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CDirectorKernel: public CKernel
{
	public:
		/** default constructor
		 *
		 */
		CDirectorKernel()
		: CKernel(), external_features(false)
		{
		}

		/**
		 */
		CDirectorKernel(bool is_external_features)
		: CKernel(), external_features(is_external_features)
		{
		}

		/** constructor
		 *
		 */
		CDirectorKernel(int32_t size, bool is_external_features)
		: CKernel(size), external_features(is_external_features)
		{
		}

		/** default constructor
		 *
		 */
		virtual ~CDirectorKernel()
		{
			cleanup();
		}

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			if (this->parallel->get_num_threads()!=1)
			{
				SG_WARNING("Enforcing to use only one thread due to restrictions of directors\n")
				this->parallel->set_num_threads(1);
			}
			return CKernel::init(l, r);
		}

		/** set the current kernel normalizer
		 *
		 * @return if successful
		 */
		virtual bool set_normalizer(CKernelNormalizer* normalizer)
		{
			return CKernel::set_normalizer(normalizer);
		}

		/** obtain the current kernel normalizer
		 *
		 * @return the kernel normalizer
		 */
		virtual CKernelNormalizer* get_normalizer()
		{
			return CKernel::get_normalizer();
		}

		/** initialize the current kernel normalizer
		 *  @return if init was successful
		 */
		virtual bool init_normalizer()
		{
			return CKernel::init_normalizer();
		}

		/** clean up your kernel
		 *
		 * base method only removes lhs and rhs
		 * overload to add further cleanup but make sure CKernel::cleanup() is
		 * called
		 */
		virtual void cleanup()
		{
			CKernel::cleanup();
		}

		virtual float64_t kernel_function(int32_t idx_a, int32_t idx_b)
		{
			SG_ERROR("Kernel function of Director Kernel needs to be overridden.\n")
			return 0;
		}

		/**
		 * get column j
		 *
		 * @return the jth column of the kernel matrix
		 */
		virtual SGVector<float64_t> get_kernel_col(int32_t j)
		{
			return CKernel::get_kernel_col(j);
		}

		/**
		 * get row i
		 *
		 * @return the ith row of the kernel matrix
		 */
		virtual SGVector<float64_t> get_kernel_row(int32_t i)
		{
			return CKernel::get_kernel_row(i);
		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return CKernel::get_num_vec_lhs();
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
		{
			return CKernel::get_num_vec_rhs();
		}

		/** set number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual void set_num_vec_lhs(int32_t num)
		{
			num_lhs=num;
		}

		/** set number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual void set_num_vec_rhs(int32_t num)
		{
			num_rhs=num;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			if (!external_features)
				return CKernel::has_features();
			else
				return true;
		}

		/** remove lhs and rhs from kernel */
		virtual void remove_lhs_and_rhs()
		{
			CKernel::remove_lhs_and_rhs();
		}

		/** remove lhs from kernel */
		virtual void remove_lhs()
		{
			CKernel::remove_lhs();
		}

		/** remove rhs from kernel */
		virtual void remove_rhs()
		{
			CKernel::remove_rhs();
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type DIRECTOR
		 */
		virtual EKernelType get_kernel_type() { return K_DIRECTOR; }

		 /** return what type of features kernel can deal with
		  *
		  * @return feature type ANY
		  */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		 /** return what class of features kernel can deal with
		  *
		  * @return feature class ANY
		  */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Director
		 */
		virtual const char* get_name() const { return "DirectorKernel"; }

		/** for optimizable kernels, i.e. kernels where the weight
		 * vector can be computed explicitly (if it fits into memory)
		 */
		virtual void clear_normal()
		{
			CKernel::clear_normal();
		}

		/** add vector*factor to 'virtual' normal vector
		 *
		 * @param vector_idx index
		 * @param weight weight
		 */
		virtual void add_to_normal(int32_t vector_idx, float64_t weight)
		{
			CKernel::add_to_normal(vector_idx, weight);
		}

		/** set optimization type
		 *
		 * @param t optimization type to set
		 */
		virtual void set_optimization_type(EOptimizationType t)
		{
			CKernel::set_optimization_type(t);
		}

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t *weights)
		{
			return CKernel::init_optimization(count, IDX, weights);
		}

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization()
		{
			return CKernel::delete_optimization();
		}

		/** compute optimized
		 *
		 * @param vector_idx index to compute
		 * @return optimized value at given index
		 */
		virtual float64_t compute_optimized(int32_t vector_idx)
		{
			return CKernel::compute_optimized(vector_idx);
		}

		/** computes output for a batch of examples in an optimized fashion
		 * (favorable if kernel supports it, i.e. has KP_BATCHEVALUATION.  to
		 * the outputvector target (of length num_vec elements) the output for
		 * the examples enumerated in vec_idx are added. therefore make sure
		 * that it is initialized with ZERO. the following num_suppvec, IDX,
		 * alphas arguments are the number of support vectors, their indices
		 * and weights
		 */
		virtual void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0)
		{
			CKernel::compute_batch(num_vec, vec_idx, target, num_suppvec, IDX, alphas, factor);
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		virtual int32_t get_num_subkernels()
		{
			return CKernel::get_num_subkernels();
		}

		/** compute by subkernel
		 *
		 * @param vector_idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		virtual void compute_by_subkernel(
			int32_t vector_idx, float64_t * subkernel_contrib)
		{
			CKernel::compute_by_subkernel(vector_idx, subkernel_contrib);
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		virtual const float64_t* get_subkernel_weights(int32_t& num_weights)
		{
			return CKernel::get_subkernel_weights(num_weights);
		}

		/** set subkernel weights
		 *
		 * @param weights new subkernel weights
		 */
		virtual void set_subkernel_weights(SGVector<float64_t> weights)
		{
			CKernel::set_subkernel_weights(weights);
		}

	protected:
		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_pre() throw (ShogunException)
		{
			CKernel::load_serializable_pre();
		}

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_post() throw (ShogunException)
		{
			CKernel::load_serializable_post();
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_pre() throw (ShogunException)
		{
			CKernel::save_serializable_pre();
		}

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_post() throw (ShogunException)
		{
			CKernel::save_serializable_post();
		}

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)
		{
			return kernel_function(idx_a, idx_b);
		}

		virtual void register_params()
		{
			CKernel::register_params();
		}

	protected:
		/* is external features */
		bool external_features;
};
}
#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTORKERNEL_H__ */
