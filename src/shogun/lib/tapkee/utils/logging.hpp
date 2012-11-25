/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_LOGGING_H_
#define TAPKEE_LOGGING_H_

#include <iostream>
#include <string>

using std::cout;
using std::ostream;
using std::string;

#define LEVEL_ENABLED_FIELD(X) bool X##_enabled
#define LEVEL_ENABLED_FIELD_INITIALIZER(X,value) X##_enabled(value)
#define LEVEL_HANDLERS(LEVEL) \
		void enable_##LEVEL() { LEVEL##_enabled = true; };		\
		void disable_##LEVEL() { LEVEL##_enabled = false; };	\
		void message_##LEVEL(const string& msg)					\
		{														\
			if (LEVEL##_enabled)								\
				impl->message_##LEVEL(msg);						\
		}
#define LEVEL_HANDLERS_DECLARATION(LEVEL) \
		virtual void message_##LEVEL(const string& msg) = 0;
#define LEVEL_HANDLERS_DEFAULT_IMPL(LEVEL) \
		virtual void message_##LEVEL(const string& msg)			\
		{														\
			if (os_ && os_->good())								\
				(*os_) << "["#LEVEL"] " << msg << "\n";			\
		}

class LoggerImplementation
{
public:
	virtual ~LoggerImplementation() {}
	LEVEL_HANDLERS_DECLARATION(info);
	LEVEL_HANDLERS_DECLARATION(warning);
	LEVEL_HANDLERS_DECLARATION(error);
	LEVEL_HANDLERS_DECLARATION(benchmark);
};

class DefaultLoggerImplementation : public LoggerImplementation
{
public:
	DefaultLoggerImplementation() : os_(&cout) {}
	virtual ~DefaultLoggerImplementation() {}
	ostream* os_;
	LEVEL_HANDLERS_DEFAULT_IMPL(info);
	LEVEL_HANDLERS_DEFAULT_IMPL(warning);
	LEVEL_HANDLERS_DEFAULT_IMPL(error);
	LEVEL_HANDLERS_DEFAULT_IMPL(benchmark)
};

class LoggingSingleton
{
	private:
		LoggingSingleton() : impl(new DefaultLoggerImplementation),
			LEVEL_ENABLED_FIELD_INITIALIZER(info,false),
			LEVEL_ENABLED_FIELD_INITIALIZER(warning,true),
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
		LEVEL_ENABLED_FIELD(error);
		LEVEL_ENABLED_FIELD(benchmark);

	public:
		static LoggingSingleton& instance()
		{
			static LoggingSingleton s;
			return s;
		}

		LEVEL_HANDLERS(info);
		LEVEL_HANDLERS(warning);
		LEVEL_HANDLERS(error);
		LEVEL_HANDLERS(benchmark);

};

#undef LEVEL_HANDLERS
#undef LEVEL_ENABLED_FIELD

#endif
