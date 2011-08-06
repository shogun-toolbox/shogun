/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <shogun/lib/vw/vw_label.h>

using namespace shogun;

void VwLabel::parse_label(v_array<substring>& words)
{
	switch(words.index())
	{
	case 0:
		break;
	case 1:
		label = float_of_substring(words[0]);
		break;
	case 2:
		label = float_of_substring(words[0]);
		weight = float_of_substring(words[1]);
		break;
	case 3:
		label = float_of_substring(words[0]);
		weight = float_of_substring(words[1]);
		initial = float_of_substring(words[2]);
		break;
	default:
		SG_SERROR("malformed example!\n"
			  "words.index() = %d\n", words.index());
	}
}
