/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Bjoern Esser
 */

#ifndef __PLIFARRAY_H__
#define __PLIFARRAY_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/DynArray.h>
#include <shogun/structure/PlifBase.h>

namespace shogun
{

/** @brief class PlifArray */
class PlifArray: public PlifBase
{
	public:
		/** default constructor */
		PlifArray();
		virtual ~PlifArray();

		/** add plif
		 *
		 * @param new_plif the new plif to be added
		 */
		void add_plif(std::shared_ptr<PlifBase> new_plif);

		/** clear */
		void clear();

		/** get number of plifs
		 *
		 * @return number of plifs
		 */
		int32_t get_num_plifs()
		{
			return m_array.get_num_elements();
		}

		/** lookup penalty float64_t
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		virtual float64_t lookup_penalty(
			float64_t p_value, float64_t* svm_values) const;

		/** lookup penalty int32_t
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		virtual float64_t lookup_penalty(
			int32_t p_value, float64_t* svm_values) const;

		/** penalty clear derivative */
		virtual void penalty_clear_derivative();

		/** penalty add derivative
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @param factor weighting the added value
		 */
		virtual void penalty_add_derivative(
			float64_t p_value, float64_t* svm_values, float64_t factor);

		/** get maximum value
		 *
		 * @return maximum value
		 */
		virtual float64_t get_max_value() const
		{
			return max_value;
		}

		/** get minimum value
		 *
		 * @return minumum value
		 */
		virtual float64_t get_min_value() const
		{
			return min_value;
		}

		/** check if plif uses SVM values
		 *
		 * @return if plif uses SVM values
		 */
		virtual bool uses_svm_values() const;

		/** get maximum ID
		 *
		 * @return maximum ID
		 */
		virtual int32_t get_max_id() const;

		void get_used_svms(int32_t* num_svms, int32_t* svm_ids);

		/** print PLIF
		 *
		 * lists all PLIFs in array
		 */
		virtual void list_plif() const
		{
			io::print("CPlifArray(num_elements={}, min_value={:1.2f}, max_value={:1.2f})\n", m_array.get_num_elements(), min_value, max_value);
			for (int32_t i=0; i<m_array.get_num_elements(); i++)
			{
				io::print("{}. ", i);
				m_array[i]->list_plif() ;
			}
		}

		/** @return object name */
		virtual const char* get_name() const { return "PlifArray"; }

	protected:
		/** plif array */
		DynArray<std::shared_ptr<PlifBase>> m_array;
		/** maximum value */
		float64_t max_value;
		/** minimum value */
		float64_t min_value;
};
}
#endif
