/** A simple formatter that uses simple "{}" placeholder.
 * Resembles SLF4J and Python format.
 *
 * Copyright (c) 2013, Sergey Lisitsyn <lisitsyn.s.o@gmail.com>
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef FORMATTING_WRAPPERS_H_
#define FORMATTING_WRAPPERS_H_

#include <limits>
#include <iomanip>

namespace formatting
{

namespace utils
{
	template <bool> struct compile_time_assert;
	template <> struct compile_time_assert<true> {};
}

namespace wrappers
{
	template <typename T>
	struct HexWrapper
	{
		utils::compile_time_assert<std::numeric_limits<T>::is_integer> HEX_USED_FOR_NON_INTEGER_TYPE;
		explicit HexWrapper(T value) : value_(value) { }
		const T value_;

		template <typename U>
		friend std::ostream& operator<<(std::ostream& out, const HexWrapper<U>& h);
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& out, const HexWrapper<T>& h)
	{
		out << "0x" << std::hex << std::uppercase << h.value_;
		return out;
	}

	template <typename T>
	struct OctWrapper
	{
		utils::compile_time_assert<std::numeric_limits<T>::is_integer> OCT_USED_FOR_NON_INTEGER_TYPE;
		explicit OctWrapper(T value) : value_(value) { }
		const T value_;

		template <typename U>
		friend std::ostream& operator<<(std::ostream& out, const HexWrapper<U>& h);
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& out, const OctWrapper<T>& h)
	{
		out << "0" << std::oct << std::uppercase << h.value_;
		return out;
	}

	template <typename T>
	struct WidthWrapper
	{
		explicit WidthWrapper(unsigned int width, T value) : value_(value), width_(width) { }
		const T value_;
		const unsigned int width_;

		template <typename U>
		friend std::ostream& operator<<(std::ostream& out, const WidthWrapper& h);
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& out, const WidthWrapper<T>& h)
	{
		out << std::setw(h.width_) << h.value_;
		return out;
	}

	struct WidthWrapperBuilder
	{
		explicit WidthWrapperBuilder(unsigned int width) : width_(width) { }
		unsigned int width_;

		template <typename T>
		inline WidthWrapper<T> operator()(T value)
		{
			return WidthWrapper<T>(width_,value);
		}
	};

	struct WidthWrapperBuilderHelper
	{
		WidthWrapperBuilderHelper() { }
		inline wrappers::WidthWrapperBuilder operator[](unsigned int w) const
		{
			return wrappers::WidthWrapperBuilder(w);
		}
	};

	template <typename T>
	struct PrecisionWrapper
	{
		utils::compile_time_assert<std::numeric_limits<T>::is_specialized> PRECISION_USED_FOR_NON_NUMERIC_TYPE;
		explicit PrecisionWrapper(unsigned int precision, T value) : value_(value), precision_(precision) { }
		const T value_;
		const unsigned int precision_;

		template <typename U>
		friend std::ostream& operator<<(std::ostream& out, const PrecisionWrapper& h);
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& out, const PrecisionWrapper<T>& h)
	{
		out << std::setprecision(h.precision_) << h.value_;
		return out;
	}

	struct PrecisionWrapperBuilder
	{
		explicit PrecisionWrapperBuilder(unsigned int precision) : precision_(precision) { }
		unsigned int precision_;

		template <typename T>
		inline PrecisionWrapper<T> operator()(T value)
		{
			return PrecisionWrapper<T>(precision_,value);
		}
	};

	struct PrecisionWrapperBuilderHelper
	{
		PrecisionWrapperBuilderHelper() { }
		inline wrappers::PrecisionWrapperBuilder operator[](unsigned int p) const
		{
			return wrappers::PrecisionWrapperBuilder(p);
		}
	};


}

/** Returns a wrapper that makes the provided
 * value represented as hex when formatting.
 *
 * E.g. formatting::hex(10) => '0xA'
 *
 * @param value a numerical value to be presented as hex
 */
template<typename T>
inline wrappers::HexWrapper<T> hex(T value)
{
	return wrappers::HexWrapper<T>(value);
}

/** Returns a wrapper that makes the provided
 * value represented as oct when formatting.
 *
 * E.g formatting::oct(9) => '011'
 *
 * @param value a numerical value to be presented as oct
 */
template<typename T>
inline wrappers::OctWrapper<T> oct(T value)
{
	return wrappers::OctWrapper<T>(value);
}

/** Returns a wrapper that makes the provided
 * pointer represented as hex value of a pointer.
 *
 * @param value a pointer to be represented as hex
 */
inline wrappers::HexWrapper<size_t> raw(void* value)
{
	size_t ptr = reinterpret_cast<size_t>(value);
	return wrappers::HexWrapper<size_t>(ptr);
}

/** Returns a wrapper that makes the provided
 * pointer represented as hex value of a pointer.
 *
 * @param value a pointer to be represented as hex
 */
inline wrappers::HexWrapper<size_t> raw(const void* value)
{
	size_t ptr = reinterpret_cast<size_t>(value);
	return wrappers::HexWrapper<size_t>(ptr);
}

/** Width wrapper helper that allows to set output width
 * with the brackets operator (e.g. width[3]('c') => "  c").
 */
static const wrappers::WidthWrapperBuilderHelper width;

/** Precision wrapper helper that allows to set output
 * precision with the brackets operator
 * (e.g. precision[6](2.718281828) => "2.71828")
 */
static const wrappers::PrecisionWrapperBuilderHelper precision;

}
#endif
