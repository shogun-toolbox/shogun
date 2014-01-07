/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIFARRAY_H__
#define __PLIFARRAY_H__

#include <lib/common.h>
#include <mathematics/Math.h>
#include <base/DynArray.h>
#include <structure/PlifBase.h>

namespace shogun
{

/** @brief class PlifArray */
class CPlifArray: public CPlifBase
{
	public:
		/** default constructor */
		CPlifArray();
		virtual ~CPlifArray();

		/** add plif
		 *
		 * @param new_plif the new plif to be added
		 */
		void add_plif(CPlifBase* new_plif);

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
			SG_PRINT("CPlifArray(num_elements=%i, min_value=%1.2f, max_value=%1.2f)\n", m_array.get_num_elements(), min_value, max_value)
			for (int32_t i=0; i<m_array.get_num_elements(); i++)
			{
				SG_PRINT("%i. ", i)
				m_array[i]->list_plif() ;
			}
		}

		/** @return object name */
		virtual const char* get_name() const { return "PlifArray"; }

	protected:
		/** plif array */
		DynArray<CPlifBase*> m_array;
		/** maximum value */
		float64_t max_value;
		/** minimum value */
		float64_t min_value;
};
}
#endif
