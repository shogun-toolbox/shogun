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