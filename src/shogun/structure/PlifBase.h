/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_BASE_H__
#define __PLIF_BASE_H__

#include <lib/common.h>
#include <base/SGObject.h>
#include <mathematics/Math.h>

namespace shogun
{
/** @brief class PlifBase */
class CPlifBase : public CSGObject
{
	public:
		/** default constructor */
		CPlifBase() {};
		virtual ~CPlifBase() {};

		/** lookup penalty float64_t
		 *
		 * abstract base method
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return penalty
		 */
		virtual float64_t lookup_penalty(
			float64_t p_value, float64_t* svm_values) const =0;

		/** lookup penalty int32_t
		 *
		 * abstract base method
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return penalty
		 */
		virtual float64_t lookup_penalty(
			int32_t p_value, float64_t* svm_values) const =0;

		/** penalty clear derivative
		 *
		 * abstrace base method
		 */
		virtual void penalty_clear_derivative()=0;

		/** penalty add derivative
		 *
		 * abstract base method
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @param factor factor weighting the added value
		 */
		virtual void penalty_add_derivative(
			float64_t p_value, float64_t* svm_values, float64_t factor)=0 ;

		/** get maximum value
		 *
		 * abstract base method
		 *
		 * @return maximum value
		 */
		virtual float64_t get_max_value() const = 0;

		/** get minimum value
		 *
		 * abstract base method
		 *
		 * @return minimum value
		 */
		virtual float64_t get_min_value() const = 0;

		/** get SVM_ids and number of SVMs used
		 *
		 * abstract base method
		 */
		virtual void get_used_svms(int32_t* num_svms, int32_t* svm_ids) = 0;

		/** if plif uses SVM values
		 *
		 * abstract base method
		 *
		 * @return if plif uses SVM values
		 */
		virtual bool uses_svm_values() const = 0;

		/** get maximum ID
		 *
		 * abstract base method
		 *
		 * @return maximum ID
		 */
		virtual int32_t get_max_id() const = 0;

		/** print PLIF
		 *
		 * abstract base method
		 */
		virtual void list_plif() const = 0 ;

};
}
#endif
