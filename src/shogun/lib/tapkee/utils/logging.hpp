/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_LOGGING_H_
#define TAPKEE_LOGGING_H_

#include <iostream>
#include <string>
#include <formatting/formatting.hpp>

#define LEVEL_ENABLED_FIELD(X) bool X##_enabled
#define LEVEL_ENABLED_FIELD_INITIALIZER(X,value) X##_enabled(value)
#define LEVEL_HANDLERS(LEVEL) \
		void enable_##LEVEL() { LEVEL##_enabled = true; };		\
		void disable_##LEVEL() { LEVEL##_enabled = false; };	\
		bool is_##LEVEL##_enabled() { return LEVEL##_enabled; };\
		void message_##LEVEL(const std::string& msg)			\
		{														\
			if (LEVEL##_enabled)								\
				impl->message_##LEVEL(msg);						\
		}
#define LEVEL_HANDLERS_DECLARATION(LEVEL) \
		virtual void message_##LEVEL(const std::string& msg) = 0
#define LEVEL_HANDLERS_DEFAULT_IMPL(STREAM, LEVEL) \
		virtual void message_##LEVEL(const std::string& msg)	\
		{														\
			if (STREAM && STREAM->good())						\
				(*STREAM) << "["#LEVEL"] " << msg << "\n";		\
		}

namespace tapkee
{

//! A base class for logger required by the library
class LoggerImplementation
{
public:
	LoggerImplementation() {};
	virtual ~LoggerImplementation() {};
	LEVEL_HANDLERS_DECLARATION(info);
	LEVEL_HANDLERS_DECLARATION(warning);
	LEVEL_HANDLERS_DECLARATION(debug);
	LEVEL_HANDLERS_DECLARATION(error);
	LEVEL_HANDLERS_DECLARATION(benchmark);
private:
	LoggerImplementation& operator=(const LoggerImplementation&);
	LoggerImplementation(const LoggerImplementation&);
};

//! Default std::cout implementation of @ref LoggerImplementation
class DefaultLoggerImplementation : public LoggerImplementation
{
public:
	DefaultLoggerImplementation() : sout(&std::cout), serr(&std::cerr) {}
	virtual ~DefaultLoggerImplementation() {}
	LEVEL_HANDLERS_DEFAULT_IMPL(serr,info);
	LEVEL_HANDLERS_DEFAULT_IMPL(serr,warning);
	LEVEL_HANDLERS_DEFAULT_IMPL(serr,debug);
	LEVEL_HANDLERS_DEFAULT_IMPL(serr,error);
	LEVEL_HANDLERS_DEFAULT_IMPL(serr,benchmark)
private:
	DefaultLoggerImplementation& operator=(const DefaultLoggerImplementation&);
	DefaultLoggerImplementation(const DefaultLoggerImplementation&);

	std::ostream* sout;
	std::ostream* serr;
};

//! Main logging singleton used by the library. Can use provided
//! @ref LoggerImplementation if necessary. By default uses
//! @ref DefaultLoggerImplementation.
class LoggingSingleton
{
	private:
		LoggingSingleton() : impl(new DefaultLoggerImplementation),
			LEVEL_ENABLED_FIELD_INITIALIZER(info,false),
			LEVEL_ENABLED_FIELD_INITIALIZER(warning,true),
			LEVEL_ENABLED_FIELD_INITIALIZER(debug,false),
			LEVEL_ENABLED_FIELD_INITIALIZER(error,true),
			LEVEL_ENABLED_FIELD_INITIALIZER(benchmark,false)
		{
		};
		~LoggingSingleton()
		{
			delete impl;
		}
		LoggingSingleton(const LoggingSingleton& ls);
		void operator=(const LoggingSingleton& ls);

		LoggerImplementation* impl;

		LEVEL_ENABLED_FIELD(info);
		LEVEL_ENABLED_FIELD(warning);
		LEVEL_ENABLED_FIELD(debug);
		LEVEL_ENABLED_FIELD(error);
		LEVEL_ENABLED_FIELD(benchmark);

	public:
		//! @return instance of the singleton
		static LoggingSingleton& instance()
		{
			static LoggingSingleton s;
			return s;
		}

		//! getter for logger implementation
		//! @return current logger implementation
		LoggerImplementation* get_logger_impl() const { return impl; }
		//! setter for logger implementation
		//! @param i logger implementation to be set
		void set_logger_impl(LoggerImplementation* i) { delete impl; impl = i; }

		LEVEL_HANDLERS(info);
		LEVEL_HANDLERS(warning);
		LEVEL_HANDLERS(debug);
		LEVEL_HANDLERS(error);
		LEVEL_HANDLERS(benchmark);

};

#undef LEVEL_HANDLERS
#undef LEVEL_HANDLERS_DECLARATION
#undef LEVEL_HANDLERS_DEFAULT_IMPL
#undef LEVEL_ENABLED_FIELD
#undef LEVEL_ENABLED_FIELD_INITIALIZER
}

#endif
