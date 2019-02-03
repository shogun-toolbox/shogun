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

#ifndef FORMATTING_IMPLEMENTATIONS_H_
#define FORMATTING_IMPLEMENTATIONS_H_

namespace formatting
{
	/** Default precision - be careful to change due to no thread safety */
	static unsigned int default_precision = 9;

	namespace internal
	{
		class ValueWrapperImplementationBase
		{
		public:
			virtual ~ValueWrapperImplementationBase() { }
			virtual std::string representation() const = 0;
		};

		template <typename T>
		class ValueWrapperImplementation :
			public ValueWrapperImplementationBase
		{
		public:
			ValueWrapperImplementation(const T& value) :
				value_(value) { }
			virtual std::string representation() const
			{
				std::stringstream string_stream;
				string_stream << std::setprecision(default_precision) << value_;
				return string_stream.str();
			}
		private:
			const T value_;
		};

		template <>
		class ValueWrapperImplementation<const char*> :
			public ValueWrapperImplementationBase
		{
		public:
			ValueWrapperImplementation(const char* value) :
				value_(value) { }
			virtual std::string representation() const
			{
				return std::string(value_);
			}
		private:
			const char* value_;
		};

		template <>
		class ValueWrapperImplementation<bool> :
			public ValueWrapperImplementationBase
		{
		public:
			ValueWrapperImplementation(bool value) :
				value_(value) { }
			virtual std::string representation() const
			{
				return value_ ? "true" : "false";
			}
		private:
			bool value_;
		};

		template <>
		class ValueWrapperImplementation<std::string> :
			public ValueWrapperImplementationBase
		{
		public:
			ValueWrapperImplementation(const std::string& value) :
				value_(value) { }
			virtual std::string representation() const
			{
				return value_;
			}
		private:
			const std::string value_;
		};

		template <typename T>
		class ValueWrapperImplementation<T*> :
			public ValueWrapperImplementationBase
		{
		public:
			ValueWrapperImplementation(const T* value) :
				value_(value) { }
			virtual std::string representation() const
			{
				std::stringstream string_stream;
				string_stream << *value_;
				return string_stream.str();
			}
		private:
			const T* value_;
		};
	}
}
#endif
