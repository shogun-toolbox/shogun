/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Viktor Gal, Giovanni De Toni, Sergey Lisitsyn,
 *          Heiko Strathmann, Yuyu Zhang, Shashwat Lal Das, Thoralf Klein,
 *          Bjoern Esser, Evangelos Anagnostopoulos, Sanuj Sharma,
 *          Saurabh Goyal
 */

#ifndef __SGIO_H__
#define __SGIO_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/io/fmt/fmt.h>

#include <string.h>
#include <locale.h>
#include <sys/types.h>
#include <type_traits>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace spdlog
{
	class logger;

	namespace details
	{
		class thread_pool;
	}

	namespace sinks
	{
		class sink;
	}
}

namespace shogun
{
/** The io libs output [DEBUG] etc in front of every message 'higher' messages
 * filter output depending on the loglevel, i.e. CRITICAL messages will print
 * all MSG_CRITICAL TO MSG_EMERGENCY messages.
 */
enum EMessageType
{
	MSG_GCDEBUG=0,
	MSG_DEBUG=1,
	MSG_INFO=2,
	MSG_WARN=3,
	MSG_ERROR=4,
	MSG_CRITICAL=5,
	MSG_MESSAGEONLY=6
};

/** The io functions can optionally prepend the function name or the line from
 * where the print occurred.
 */
enum EMessageLocation
{
	MSG_NONE=0,
	MSG_FUNCTION=1,
	MSG_LINE_AND_FILE=2
};

#ifdef DARWIN
#include <Availability.h>
#endif //DARWIN

#define SG_SET_LOCALE_C setlocale(LC_ALL, "C")
#define SG_RESET_LOCALE setlocale(LC_ALL, "")

#if !defined(SG_UNLIKELY)
#if __GNUC__ >= 3
#define SG_UNLIKELY(expr) __builtin_expect(expr, 0)
#else
#define SG_UNLIKELY(expr) expr
#endif
#endif

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

// printf like functions (with additional severity level)
#define SG_IO env()->io()
#define _SRC_LOC SourceLocation{__FILE__, __LINE__, __PRETTY_FUNCTION__}

#define SG_GCDEBUG(...) {								\
	SG_IO->message(MSG_GCDEBUG, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_DEBUG(...) {									\
	SG_IO->message(MSG_DEBUG, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_INFO(...) {									\
	SG_IO->message(MSG_INFO, _SRC_LOC, __VA_ARGS__);		\
}

#define SG_WARNING(...) {							\
	SG_IO->message(MSG_WARN, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_THROW(ExceptionType, ...)                                           \
	{                                                                          \
		SG_IO->template error<ExceptionType>(                                  \
		    MSG_ERROR, _SRC_LOC, __VA_ARGS__);  \
	}
#define SG_ERROR(...) SG_THROW(ShogunException, __VA_ARGS__)

#define SG_UNSTABLE(func, ...) { SG_IO->message(MSG_WARN, _SRC_LOC, \
__FILE__ ":" func ": Unstable method!  Please report if it seems to " \
"work or not to the Shogun mailing list.  Thanking you in " \
"anticipation.  " __VA_ARGS__); }

#define SG_PRINT(...) { SG_IO->message(MSG_MESSAGEONLY, _SRC_LOC, __VA_ARGS__); }
#define SG_NOTIMPLEMENTED { SG_IO->not_implemented(_SRC_LOC); }
#define SG_GPL_ONLY { SG_IO->gpl_only(_SRC_LOC); }

#define SG_DONE() {								\
	if (SG_UNLIKELY(SG_IO->get_show_progress()))	\
		SG_IO->done();								\
}

// printf like function using the global SG_IO object

#define ASSERT(x) {																	\
	if (SG_UNLIKELY(!(x)))																\
		SG_ERROR("assertion {} failed in {} file {} line {}\n",#x, __PRETTY_FUNCTION__, __FILE__, __LINE__)	\
}

#define REQUIRE(x, ...) {		\
	if (SG_UNLIKELY(!(x)))		\
		SG_ERROR(__VA_ARGS__)	\
}

#define REQUIRE_E(x, Exception, ...)                                           \
	{                                                                          \
		if (SG_UNLIKELY(!(x)))                                                 \
			SG_THROW(Exception, __VA_ARGS__)                                  \
	}

/* help clang static analyzer to identify custom assertation functions */
#ifdef __clang_analyzer__
void _clang_fail(void) __attribute__((analyzer_noreturn));

#undef SG_ERROR(...)
#undef SG_ERROR(...)
#define SG_ERROR(...) _clang_fail();
#define SG_ERROR(...) _clang_fail();

#endif /* __clang_analyzer__ */

/**
 * @brief struct Substring, specified by
 * start position and end position.
 *
 * Used to mark strings in a buffer, where they
 * need not be delimited by NUL characters.
 */
struct substring
{
	/** start */
	char *start;
	/** end */
	char *end;
};

struct SourceLocation
{
	constexpr SourceLocation(
		const char* file_="", int32_t line_=0, const char* function_="")
	: file(file_), line(line_), function(function_)
	{
	}

	const char* file;
	int32_t line;
	const char* function;
};

/** @brief Class SGIO, used to do input output operations throughout shogun.
 *
 * Any debug or error or progress message is passed through the functions of
 * this class to be in the end written to the screen. Note that messages don't
 * have to be written to stdout or stderr, but can be redirected to a file.
 */
class SGIO
{
	public:
		/** default constructor */
		SGIO();

		/** destructor */
		virtual ~SGIO();

		/** (re)initializes the default (asynchronous) logger
		 * 
		 * @param queue_size size of the message queue
		 * @param n_threads number of logging threads
		 */
		void init_default_logger(uint64_t queue_size=128, uint64_t n_threads=1);

		/** (re)initializes the default sink
		 */
		void init_default_sink();

		/** set loglevel
		 *
		 * @param level level of log messages
		 */
		void set_loglevel(EMessageType level);

		/** @return whether loglevel is above specified level and thus the
		 * message should be printed
		 */
		bool should_log(EMessageType prio) const;

		/** get loglevel
		 *
		 * @return level of log messages
		 */
		EMessageType get_loglevel() const;

		/** get show_progress
		 *
		 * @return if progress bar is shown
		 */
		inline bool get_show_progress() const
		{
			return show_progress;
		}

		/** show location where printing occurs
		 */
		inline EMessageLocation get_location_info() const
		{
			return location_info;
		}

		/** get syntax highlight
		 *
		 * @return if syntax highlighting is enabled
		 */
		inline bool get_syntax_highlight() const
		{
			return syntax_highlight;
		}

		/** format and print a message
		 * @param prio message priority
		 * @param loc source code location
		 * @param msg message format
		 * @param args arguments for formatting message
		 */
		template <typename... Args>
		void message(EMessageType prio, const SourceLocation& loc, const char* fmt, const Args&... args) const;

		/** format and print a message
		 * @param prio message priority
		 * @param msg message format
		 * @param args arguments for formatting message
		 */
		template <typename... Args>
		void message(EMessageType prio, const char* fmt, const Args&... args) const
		{
			message(prio, SourceLocation{}, fmt, args...);
		}

		/** format and print a message, and then throw an exception
		 * @tparam ExceptionType type of the exception to throw
		 * @param prio message priority
		 * @param loc source code location
		 * @param args arguments for formatting message
		 */
		template <typename Exception, typename... Args>
		void error(EMessageType prio, const SourceLocation& loc, const char* fmt, const Args&... args) const;

		/** format and print a message, and then throw an exception
		 * @tparam ExceptionType type of the exception to throw
		 * @param prio message priority
		 * @param loc source code location
		 * @param args arguments for formatting message
		 */
		template <typename Exception, typename... Args>
		void error(EMessageType prio, const char* msg, const Args&... args) const
		{
			error(prio, SourceLocation{}, msg, args...);
		}

		/** print 'done' with priority INFO,
		 * but only if progress bar is enabled
		 *
		 */
		void done();

		/** print error message 'not implemented' */
		inline void not_implemented(const SourceLocation& loc={}) const
		{
			error<ShogunException>(
			    MSG_ERROR, loc, "Sorry, not yet implemented .\n");
		}

		/** print error message 'Only available with GPL parts.' */
		inline void gpl_only(const SourceLocation& loc={}) const
		{
			error<ShogunException>(
			    MSG_ERROR, loc,
			    "This feature is only "
			    "available if Shogun is built "
			    "with GPL codes.\n");
		}

		/** print warning message 'function deprecated' */
		inline void deprecated(const SourceLocation& loc={}) const
		{
			message(MSG_WARN, loc,
					"This function is deprecated and will be removed soon.\n");
		}

		/** redirects stdout to another sink */
		void redirect_stdout(std::shared_ptr<spdlog::sinks::sink> sink);

		/** redirects stderr to another sink */
		void redirect_stderr(std::shared_ptr<spdlog::sinks::sink> sink);

		/** enable progress bar */
		inline void enable_progress()
		{
			show_progress=true;
		}

		/** disable progress bar */
		inline void disable_progress()
		{
			show_progress=false;
		}

		/** enable displaying of file and line when printing messages etc
		 *
		 * @param location location info (none, function, ...)
		 *
		 */
		inline void set_location_info(EMessageLocation location)
		{
			location_info = location;
			update_pattern();
		}

		/** enable syntax highlighting */
		inline void enable_syntax_highlighting()
		{
			syntax_highlight=true;
			update_pattern();
		}

		/** disable syntax highlighting */
		inline void disable_syntax_highlighting()
		{
			syntax_highlight=false;
			update_pattern();
		}

		/**
		 * Return a C string from the substring
		 * @param s substring
		 * @return new C string representation
		 */
		static char* c_string_of_substring(substring s);

		/**
		 * Print the substring
		 * @param s substring
		 */
		static void print_substring(substring s);

		/**
		 * Get value of substring as float
		 * (if possible)
		 * @param s substring
		 * @return float32_t value of substring
		 */
		static float32_t float_of_substring(substring s);

		/**
		 * Return value of substring as double
		 * @param s substring
		 * @return substring as double
		 */
		static float64_t double_of_substring(substring s);

		/**
		 * Integer value of substring
		 * @param s substring
		 * @return int value of substring
		 */
		static int32_t int_of_substring(substring s);

		/**
		 * Unsigned long value of substring
		 * @param s substring
		 * @return unsigned long value of substring
		 */
		static uint32_t ulong_of_substring(substring s);

		/**
		 * Length of substring
		 * @param s substring
		 * @return length of substring
		 */
		static uint32_t ss_length(substring s);

		/** @return object name */
		inline const char* get_name() { return "SGIO"; }

	private:
		/** Prints a formatted message */
		void message_(EMessageType prio, const SourceLocation& loc, const fmt::string_view& msg) const;

		/** Updates log pattern */
		void update_pattern();

		/** if progress bar shall be shown */
		bool show_progress;
		/** if each print function should append filename and linenumber of
		 * where the print occurs etc */
		EMessageLocation location_info;
		/** whether syntax highlighting is enabled */
		bool syntax_highlight;

		class RedirectSink;
		std::shared_ptr<RedirectSink> io_sink;
		std::shared_ptr<spdlog::logger> io_logger;
		std::shared_ptr<spdlog::details::thread_pool> thread_pool;
};

template <typename... Args>
void SGIO::message(EMessageType prio, const SourceLocation& loc, const char* fmt, const Args&... args) const
{
	if(should_log(prio))
	{
		fmt::memory_buffer msg;
		fmt::format_to(msg, fmt, args...);
		message_(prio, loc, fmt::string_view(msg.data(), msg.size()));
	}
}

template <typename ExceptionType, typename... Args>
void SGIO::error(EMessageType prio, const SourceLocation& loc, const char* msg, const Args&... args) const
{
	static_assert(std::is_nothrow_copy_constructible<ExceptionType>::value,
			  "ExceptionType must be nothrow copy constructible");
	message(prio, loc, msg, args...);
	throw ExceptionType(msg);
}
}
#endif // __SGIO_H__
