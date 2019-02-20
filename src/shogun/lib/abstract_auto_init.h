/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ABSTRACT_AUTO_INIT_H
#define SHOGUN_ABSTRACT_AUTO_INIT_H

#include <string>

namespace shogun
{
	class Any;

	namespace factory
	{
		class AutoInit
		{
		public:
			AutoInit(const std::string& name, const std::string& description)
			    : m_name(name), m_description(description)
			{
			}

			virtual Any operator()() = 0;
			virtual ~AutoInit() = default;

		protected:
			const std::string m_name;
			const std::string m_description;
		};
	} // namespace factory
} // namespace shogun

#endif // SHOGUN_ABSTRACT_AUTO_INIT_H
