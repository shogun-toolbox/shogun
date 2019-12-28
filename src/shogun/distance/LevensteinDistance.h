/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#ifndef _LEVENSTEINDISTANCE_H__
#define _LEVENSTEINDISTANCE_H__

#include <string>
#include <vector>
namespace shogun
{
	class LevensteinDistance
	{
		/** default constructor */
	public:
		LevensteinDistance();

		/** constructor
		 *
		 * @param lhs class name of left-hand side
		 * @param rhs class name of right-hand side
		 *
		 */
		LevensteinDistance(const std::string& lhs, const std::string& rhs);

		/** destructor */
		virtual ~LevensteinDistance();

		/** get levenstein distance of two class names given in constructor
		 *
		 * @return distance of two class names given in constructor
		 */
		size_t distance();
		/** get levenstein distance of two names given in constructor
		 *
		 * @param lhs class name of left-hand side
		 * @param rhs class name of right-hand side
		 * @return distance of two names given in parameters
		 */
		size_t distance(const std::string& lhs, const std::string& rhs);

	protected:
		// compute the distance
		size_t compute(const std::string& lhs, const std::string& rsh);

	private:
		// name of left hand side
		std::string m_lhs_name;
		// name of right hand side
		std::string m_rhs_name;
	};
} // namespace shogun

#endif /* _LEVENSTEINDISTANCE_H__ */