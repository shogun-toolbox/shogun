
#ifndef __INTRON_LIST__
#define __INTRON_LIST__

#include "lib/common.h"
#include "base/SGObject.h"

/** @brief class IntronList */
class CIntronList : public CSGObject 
{
	public:
		CIntronList();
		virtual ~CIntronList();

		/** initialize all arrays with the number of candidate positions */
		void init_list(int32_t* all_pos, int32_t len);	

		/** read introns */
		void read_introns(int32_t* start_pos, int32_t* end_pos, int32_t* quality, int32_t len);

		/** get coverage and quality score */
		void get_coverage(int32_t* coverage, int32_t* quality, int32_t from_pos, int32_t to_pos);
	
		inline virtual const char* get_name() const { return "IntronList"; }
	protected:
		int32_t m_length;
		int32_t* m_all_pos;
		int32_t** m_intron_list;
		int32_t** m_quality_list;
};
#endif
