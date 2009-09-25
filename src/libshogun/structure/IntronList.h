/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Jonas Behr
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
		void get_intron_support(int32_t* values, int32_t from_pos, int32_t to_pos);
	
		inline virtual const char* get_name() const { return "IntronList"; }
	protected:
		int32_t m_length;
		int32_t* m_all_pos;
		int32_t** m_intron_list;
		int32_t** m_quality_list;
};
#endif
