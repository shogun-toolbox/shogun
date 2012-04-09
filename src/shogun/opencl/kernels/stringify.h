/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* Written (W) 2012 Philippe Tillet
*/

#ifndef STRINGIFY_H
#define STRINGIFY_H

	#ifdef __STDC__
	#define __stringify_1(x...)	#x
	#define __stringify(x...)	__stringify_1(x)
	#else
	#define __stringify_1(x)	#x
	#define __stringify(x)	 __stringify_1(x)
	#endif

#endif