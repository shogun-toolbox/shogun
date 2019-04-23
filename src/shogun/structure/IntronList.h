/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang
 */

#ifndef __INTRON_LIST__
#define __INTRON_LIST__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
/** @brief class IntronList */
class IntronList : public SGObject
{
	public:

		/** constructor
		 */
		IntronList();

		virtual ~IntronList();

		/** initialize all arrays with the number of candidate positions
		 *
		 * @param all_pos list of candidate positions
		 * @param len number of candidate positions
		 */
		void init_list(int32_t* all_pos, int32_t len);

		/** read introns
		 *
		 * @param start_pos array of start positions
		 * @param end_pos array of end positions
		 * @param quality quality scores for introns in list
		 * @param len number of items in all three previous arguments
		 */
		void read_introns(int32_t* start_pos, int32_t* end_pos, int32_t* quality, int32_t len);

		/** get coverage and quality score
		 *
		 * @param values values[0]: coverage of that intron; values[1]: associated quality score
		 * @param from_pos start position of intron
		 * @param to_pos end position of intron
		 */
		void get_intron_support(int32_t* values, int32_t from_pos, int32_t to_pos);

		/**
		 * @return object name
		 */
		virtual const char* get_name() const { return "IntronList"; }
	protected:
		/** number of positions */
		int32_t m_length;

		/** index of positions in the DNA sequence*/
		int32_t* m_all_pos;

		/** data structure storing the introns;
		 *  for all posible end positions there is a
		 *  list of start positions stored
		 */
		int32_t** m_intron_list;

		/** data structure storing the intron quality scores;
		 *  the shape is exactly the same as for the introns
		 */
		int32_t** m_quality_list;
};
}
#endif
