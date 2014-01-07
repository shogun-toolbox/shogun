/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef _MULTICLASSOCAS_H___
#define _MULTICLASSOCAS_H___

#include <lib/common.h>
#include <features/DotFeatures.h>
#include <lib/external/libocas.h>
#include <machine/LinearMulticlassMachine.h>

namespace shogun
{

/** @brief multiclass OCAS wrapper */
class CMulticlassOCAS : public CLinearMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor  */
		CMulticlassOCAS();

		/** standard constructor
		 * @param C C regularication constant value
		 * @param features features
		 * @param labs labels
		 */
		CMulticlassOCAS(float64_t C, CDotFeatures* features, CLabels* labs);

		/** destructor */
		virtual ~CMulticlassOCAS();

		/** get name */
		virtual const char* get_name() const
		{
			return "MulticlassOCAS";
		}

		/** set C
		 * @param C C value
		 */
		inline void set_C(float64_t C)
		{
			ASSERT(C>0)
			m_C = C;
		}
		/** get C
		 * @return C value
		 */
		inline float64_t get_C() const { return m_C; }

		/** set epsilon
		 * @param epsilon epsilon value
		 */
		inline void set_epsilon(float64_t epsilon)
		{
			ASSERT(epsilon>0)
			m_epsilon = epsilon;
		}
		/** get epsilon
		 * @return epsilon value
		 */
		inline float64_t get_epsilon() const { return m_epsilon; }

		/** set max iter
		 * @param max_iter max iter value
		 */
		inline void set_max_iter(int32_t max_iter)
		{
			ASSERT(max_iter>0)
			m_max_iter = max_iter;
		}
		/** get max iter
		 * @return max iter value
		 */
		inline int32_t get_max_iter() const { return m_max_iter; }

		/** set method
		 * @param method method value
		 */
		inline void set_method(int32_t method)
		{
			ASSERT(method==0 || method==1)
			m_method = method;
		}
		/** get method
		 * @return method value
		 */
		inline int32_t get_method() const { return m_method; }

		/** set buf size
		 * @param buf_size buf size value
		 */
		inline void set_buf_size(int32_t buf_size)
		{
			ASSERT(buf_size>0)
			m_buf_size = buf_size;
		}
		/** get buf size
		 * @return buf_size value
		 */
		inline int32_t get_buf_size() const { return m_buf_size; }

protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data = NULL);

		/** update W */
		static float64_t msvm_update_W(float64_t t, void* user_data);

		/** full compute W */
		static void msvm_full_compute_W(float64_t *sq_norm_W, float64_t *dp_WoldW,
		                                float64_t *alpha, uint32_t nSel, void* user_data);

		/** full add new cut */
		static int msvm_full_add_new_cut(float64_t *new_col_H, uint32_t *new_cut,
		                                 uint32_t nSel, void* user_data);

		/** full compute output */
		static int msvm_full_compute_output(float64_t *output, void* user_data);

		/** sort */
		static int msvm_sort_data(float64_t* vals, float64_t* data, uint32_t size);

		/** print nothing */
		static void msvm_print(ocas_return_value_T value);

private:

		/** register parameters */
		void register_parameters();

protected:

		/** regularization constant for each machine */
		float64_t m_C;

		/** tolerance */
		float64_t m_epsilon;

		/** max number of iterations */
		int32_t m_max_iter;

		/** method */
		int32_t m_method;

		/** buffer size */
		int32_t m_buf_size;
};
}
#endif
