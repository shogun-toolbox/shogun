/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2017 - Viktor Gal
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef __CPU_INFO_H__
#define __CPU_INFO_H__

#include <shogun/lib/common.h>

SG_FORCED_INLINE static void CpuRelax()
{
#ifdef _MSC_VER
	_mm_pause();
#elif defined(__i386__) || defined(__x86_64__)
	asm volatile("pause");
#elif defined(__arm__) || defined(__aarch64__)
	asm volatile("yield");
#elif defined(__powerpc__) || defined(__ppc__)
	asm volatile("or 27,27,27");
#elif defined(__s390__) || defined(__s390x__)
	asm volatile("" : : : "memory");
#else
#warning "Unknown architecture, defaulting to delaying loop."
	static uint32_t bar = 13;
	static uint32_t* foo = &bar;
	for (unsigned int i = 0; i < 100000; i++)
	{
		*foo = (*foo * 33) + 17;
	}
#endif
}

#endif /* __CPU_INFO_H__ */
