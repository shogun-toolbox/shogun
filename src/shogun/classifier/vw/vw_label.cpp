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

#include <classifier/vw/vw_label.h>

using namespace shogun;

void VwLabel::label_from_substring(v_array<substring>& words)
{
	switch(words.index())
	{
	case 0:
		break;
	case 1:
		label = SGIO::float_of_substring(words[0]);
		break;
	case 2:
		label = SGIO::float_of_substring(words[0]);
		weight = SGIO::float_of_substring(words[1]);
		break;
	case 3:
		label = SGIO::float_of_substring(words[0]);
		weight = SGIO::float_of_substring(words[1]);
		initial = SGIO::float_of_substring(words[2]);
		break;
	default:
		SG_SERROR("malformed example!\n"
			  "words.index() = %d\n", words.index());
	}
}
