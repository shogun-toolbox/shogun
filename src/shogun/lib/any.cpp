/*
 * Copyright (c) 2018, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2018 Heiko Strathmann
 */

#include <shogun/lib/any.h>
#include <shogun/mathematics/Math.h>
#ifdef HAVE_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace shogun
{
	namespace any_detail {
		std::string demangled_type_helper(const char *name) {
#ifdef HAVE_CXA_DEMANGLE
			size_t length;
			int status;
			char *demangled = abi::__cxa_demangle(name, nullptr, &length, &status);
			std::string demangled_string(demangled);
			free(demangled);
#else
			std::string demangled_string(name);
#endif
			return demangled_string;
		}
	}

	namespace any_detail
	{

#ifndef REAL_COMPARE_IMPL
#define REAL_COMPARE_IMPL(real_t)                                              \
	template <>                                                                \
	bool compare_impl_eq(const real_t& lhs, const real_t& rhs)                 \
	{                                                                          \
		SG_SDEBUG("Comparing using fequals<" #real_t ">(lhs, rhs).\n");        \
		return Math::fequals(                                                 \
		    lhs, rhs, std::numeric_limits<real_t>::epsilon());                 \
	}

		REAL_COMPARE_IMPL(float32_t)
		REAL_COMPARE_IMPL(float64_t)
		REAL_COMPARE_IMPL(floatmax_t)
#undef REAL_COMPARE_IMPL
#endif // REAL_COMPARE_IMPL

		template <>
		bool compare_impl_eq(const complex128_t& lhs, const complex128_t& rhs)
		{
			SG_SDEBUG("Comparing using fequals<complex128_t>(lhs, rhs).\n");
			return Math::fequals(lhs.real(), rhs.real(), LDBL_EPSILON) &&
			       Math::fequals(lhs.imag(), rhs.imag(), LDBL_EPSILON);
		}

		void free_object(SGObject* obj)
		{
			//FIXME
			//SG_UNREF(obj);
		}
	}
}
