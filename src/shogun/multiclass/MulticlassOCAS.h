#ifndef _MULTICLASSOCAS_H___
#define _MULTICLASSOCAS_H___

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/lib/external/libocas.h>
#include <shogun/machine/LinearMulticlassMachine.h>

namespace shogun
{

/** @brief multiclass OCAS wrapper */
class MulticlassOCAS : public LinearMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor  */
		MulticlassOCAS();

		/** standard constructor
		 * @param C C regularication constant value
		 * @param features features
		 * @param labs labels
		 */
		MulticlassOCAS(float64_t C, const std::shared_ptr<Features>& features, std::shared_ptr<Labels> labs);

		/** destructor */
		~MulticlassOCAS() override;

		/** get name */
		const char* get_name() const override
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
		bool train_machine(std::shared_ptr<Features> data = NULL) override;

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
