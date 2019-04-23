/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */

#ifndef SHOGUN_OBSERVEDVALUE_H
#define SHOGUN_OBSERVEDVALUE_H

#include <shogun/base/SGObject.h>

namespace shogun
{

	template <class T>
	class ObservedValueTemplated;

	/**
	* Observed value which is emitted by algorithms.
	*/
	class ObservedValue : public SGObject
	{
	public:
		/**
		 * Constructor
		 * @param step step
		 * @param name name of the observed value
		 */
		ObservedValue(const int64_t step, const std::string& name);

		/**
		 * Destructor
		 */
		~ObservedValue(){};

#ifndef SWIG

		/**
		* Return a any version of the stored type.
		* @return the any value.
		*/
		virtual const Any& get_any() const
		{
			return m_any_value;
		}

#endif

		/** @return object name */
		virtual const char* get_name() const
		{
			return "ObservedValue";
		}

	protected:
		/** ObservedValue step (used by Tensorboard to print graphs) */
		int64_t m_step;
		/** Parameter's name */
		std::string m_name;
		/** Untyped value */
		Any m_any_value;
	};
}

#endif // SHOGUN_OBSERVEDVALUE_H
