/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ABSTRACT_AUTO_INIT_H
#define SHOGUN_ABSTRACT_AUTO_INIT_H

#include <string_view>

namespace shogun
{
	class Any;

	namespace params
	{
		class AutoInit
		{
		public:
			constexpr AutoInit(std::string_view name, std::string_view description)
			    : m_name(name), m_description(description)
			{
			}
			virtual ~AutoInit() = default;
			virtual Any operator()() const = 0;

		protected:
			std::string_view m_name;
			std::string_view m_description;
		};
	} // namespace factory
} // namespace shogun

#endif // SHOGUN_ABSTRACT_AUTO_INIT_H
