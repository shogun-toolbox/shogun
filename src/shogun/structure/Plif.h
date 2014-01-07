/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_H__
#define __PLIF_H__

#include <lib/common.h>
#include <lib/SGVector.h>
#include <mathematics/Math.h>
#include <structure/PlifBase.h>

namespace shogun
{

/** Ways to transform inputs */
enum ETransformType
{
	/// Linear
	T_LINEAR,
	/// Logarithmic
	T_LOG,
	/// Logarithmic (log(1+x))
	T_LOG_PLUS1,
	/// Logarithmic (log(3+x))
	T_LOG_PLUS3,
	/// Linear (3+x)
	T_LINEAR_PLUS3
};

/** @brief class Plif */
class CPlif: public CPlifBase
{
	public:
		/** constructor
		 *
		 * @param len len
		 */
		CPlif(int32_t len=0);
		virtual ~CPlif();

		/** init penalty struct cache */
		void init_penalty_struct_cache();

		/** lookup penalty SVM
		 *
		 * @param p_value value
		 * @param d_values d values
		 * @return the penalty
		 */
		float64_t lookup_penalty_svm(
			float64_t p_value, float64_t *d_values) const;

		/** lookup penalty float64_t
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return the penalty
		 */
		float64_t lookup_penalty(
			float64_t p_value, float64_t* svm_values) const;

		/** lookup penalty int32_t
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return the penalty
		 */
		float64_t lookup_penalty(int32_t p_value, float64_t* svm_values) const;

		/** lookup
		 *
		 * @param p_value value
		 * @return a penalty
		 */
		inline float64_t lookup(float64_t p_value)
		{
			ASSERT(use_svm == 0)
			return lookup_penalty(p_value, NULL);
		}

		/** penalty clear derivative */
		void penalty_clear_derivative();

		/** penalty add derivative SVM
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @param factor factor weighting the added value
		 */
		void penalty_add_derivative_svm(
			float64_t p_value, float64_t* svm_values, float64_t factor) ;

		/** penalty add derivative
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @param factor factor weighting the added value
		 */
		void penalty_add_derivative(float64_t p_value, float64_t* svm_values, float64_t factor);

		/** get cum derivative
		 *
		 * @param p_len len
		 * @return cum derivative
		 */
		const float64_t * get_cum_derivative(int32_t & p_len) const
		{
			p_len = len;
			return cum_derivatives.vector;
		}

		/** set transform type
		 *
		 * @param type_str type (string)
		 * @return if setting was successful
		 */
		bool set_transform_type(const char *type_str);

		/** get transform type
		 *
		 * @return type_str type (string)
		 */
		const char* get_transform_type()
		{
			if (transform== T_LINEAR)
				return "linear";
			else if (transform== T_LOG)
				return "log";
			else if (transform== T_LOG_PLUS1)
				return "log(+1)";
			else if (transform== T_LOG_PLUS3)
				return "log(+3)";
			else if (transform== T_LINEAR_PLUS3)
				return "(+3)";
			else
				SG_ERROR("wrong type")
			return "";
		}


		/** set ID
		 *
		 * @param p_id the id to set
		 */
		void set_id(int32_t p_id)
		{
			id=p_id;
		}

		/** get ID
		 *
		 * @return the ID
		 */
		int32_t get_id() const
		{
			return id;
		}

		/** get maximum ID
		 *
		 * @return maximum ID
		 */
		int32_t get_max_id() const
		{
			return get_id();
		}

		/** set use SVM
		 *
		 * @param p_use_svm if SVM shall be used
		 */
		void set_use_svm(int32_t p_use_svm)
		{
			invalidate_cache();
			use_svm=p_use_svm;
		}

		/** get use SVM
		 *
		 * @return if SVM is used
		 */
		int32_t get_use_svm() const
		{
			return use_svm;
		}

		/** check if plif uses SVM values
		 *
		 * @return if plif uses SVM values
		 */
		virtual bool uses_svm_values() const
		{
			return (get_use_svm()!=0);
		}

		/** set use cache
		 *
		 * @param p_use_cache if cache shall be used
		 */
		void set_use_cache(int32_t p_use_cache)
		{
			invalidate_cache();
			use_cache=p_use_cache;
		}

		/** invalidate the cache
		 */
		void invalidate_cache()
		{
			SG_FREE(cache);
			cache=NULL;
		}

		/** get use cache
		 *
		 * @return if cache is used
		 */
		int32_t get_use_cache()
		{
			return use_cache;
		}

		/** set plif
		 *
		 * @param p_len len
		 * @param p_limits limit
		 * @param p_penalties penalties
		 */
		void set_plif(
			int32_t p_len, float64_t *p_limits, float64_t* p_penalties)
		{
			ASSERT(len==p_len)

			for (int32_t i=0; i<len; i++)
			{
				limits[i]=p_limits[i];
				penalties[i]=p_penalties[i];
			}

			invalidate_cache();
			penalty_clear_derivative();
		}

		/** set plif_limits
		 *
		 * @param p_limits limit
		 */
		void set_plif_limits(SGVector<float64_t> p_limits)
		{
			ASSERT(len==p_limits.vlen)

			limits = p_limits;

			invalidate_cache();
			penalty_clear_derivative();
		}


		/** set plif penalty
		 *
		 * @param p_penalties penalties
		 */
		void set_plif_penalty(SGVector<float64_t> p_penalties)
		{
			ASSERT(len==p_penalties.vlen)

			penalties = p_penalties;

			invalidate_cache();
			penalty_clear_derivative();
		}

		/** set plif length
		 *
		 * @param p_len len
		 */
		void set_plif_length(int32_t p_len)
		{
			if (len!=p_len)
			{
				len=p_len;

				SG_DEBUG("set_plif len=%i\n", p_len)
				limits = SGVector<float64_t>(len);
				penalties = SGVector<float64_t>(len);
				cum_derivatives = SGVector<float64_t>(len);
			}

			for (int32_t i=0; i<len; i++)
			{
				limits[i]=0.0;
				penalties[i]=0.0;
				cum_derivatives[i]=0.0;
			}

			invalidate_cache();
			penalty_clear_derivative();
		}

		/** get Plif limits
		 *
		 * @return limits
		 */
		SGVector<float64_t> get_plif_limits()
		{
			return limits;
		}

		/** get plif penalty
		 *
		 * @return plif penalty
		 */
		SGVector<float64_t> get_plif_penalties()
		{
			return penalties;
		}

		/** set maximum value
		 *
		 * @param p_max_value maximum value
		 */
		inline void set_max_value(float64_t p_max_value)
		{
			max_value=p_max_value;
			invalidate_cache();
		}

		/** get maximum value
		 *
		 * @return maximum value
		 */
		virtual float64_t get_max_value() const
		{
			return max_value;
		}

		/** set minimum value
		 *
		 * @param p_min_value minimum value
		 */
		inline void set_min_value(float64_t p_min_value)
		{
			min_value=p_min_value;
			invalidate_cache();
		}

		/** get minimum value
		 *
		 * @return minimum value
		 */
		virtual float64_t get_min_value() const
		{
			return min_value;
		}

		/** set name
		 *
		 * @param p_name name
		 */
		void set_plif_name(char *p_name);

		/** get name
		 *
		 * @return name
		 */
		char* get_plif_name() const;

		/** get do calc
		 *
		 * @return if calc shall be done
		 */
		bool get_do_calc();

		/** set do calc
		 *
		 * @param b if calc shall be done
		 */
		void set_do_calc(bool b);

		/** get SVM_ids and number of SVMs used
		 *
		 */
		void get_used_svms(int32_t* num_svms, int32_t* svm_ids);

		/** get plif len
		 *
		 * @return plif len
		 */
		inline int32_t get_plif_len()
		{
			return len;
		}

		/** print PLIF
		 *
		 * lists some properties of the PLIF
		 */
		virtual void list_plif() const
		{
			SG_PRINT("CPlif(min_value=%1.2f, max_value=%1.2f, use_svm=%i)\n", min_value, max_value, use_svm)
		}

		/** delete plif struct
		 *
		 * @param PEN array of plifs
		 * @param P id of plif
		 */
		static void delete_penalty_struct(CPlif** PEN, int32_t P);

		/** @return object name */
		virtual const char* get_name() const { return "Plif"; }

	protected:
		/** len */
		int32_t len;
		/** limits */
		SGVector<float64_t> limits;
		/** penalties */
		SGVector<float64_t> penalties;
		/** cum derivatives */
		SGVector<float64_t> cum_derivatives;
		/** maximum value */
		float64_t max_value;
		/** minimum value */
		float64_t min_value;
		/** cache */
		float64_t *cache;
		/** transform type */
		enum ETransformType transform;
		/** id */
		int32_t id;
		/** name */
		char * name;
		/** if SVM shall be used */
		int32_t use_svm;
		/** if cache shall be used */
		bool use_cache;
		/** do calc
		 *  if this is true: lookup_penalty behaves normal
		 *  else: lookup_penalty returns the p_value untransformed*/
		bool do_calc;
};
}
#endif
