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

#include <string.h>
#include <locale.h>
#include <sys/types.h>
#include <functional>
#include <type_traits>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace shogun
{
	class RefCount;
	class SGIO;
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

#define NUM_LOG_LEVELS 7
#define FBUFSIZE 4096

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
// for object derived from CSGObject
#define _SRC_LOC spdlog::source_loc{__FILE__, __LINE__, __PRETTY_FUNCTION__}

#define SG_GCDEBUG(...) {								\
	io->message(MSG_GCDEBUG, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_DEBUG(...) {									\
	io->message(MSG_DEBUG, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_OBJ_DEBUG(o,...) {							\
	o->io->message(MSG_DEBUG, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_INFO(...) {									\
	io->message(MSG_INFO, _SRC_LOC, __VA_ARGS__);		\
}

#define SG_CLASS_INFO(c, ...) {							\
	c::io->message(MSG_INFO, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_WARNING(...) { io->message(MSG_WARN, _SRC_LOC, __VA_ARGS__); }
#define SG_THROW(ExceptionType, ...)                                           \
	{                                                                          \
		io->template error<ExceptionType>(                                     \
		    MSG_ERROR, _SRC_LOC, __VA_ARGS__);  \
	}
#define SG_ERROR(...) SG_THROW(ShogunException, __VA_ARGS__)
#define SG_OBJ_ERROR(o, ...)                                                   \
	{                                                                          \
		o->io->template error<ShogunException>(                                \
		    MSG_ERROR, _SRC_LOC, __VA_ARGS__);  \
	}
#define SG_CLASS_ERROR(c, ...)                                                 \
	{                                                                          \
		c::io->template error<ShogunException>(                                \
		    MSG_ERROR, _SRC_LOC, __VA_ARGS__);  \
	}
#define SG_UNSTABLE(func, ...) { io->message(MSG_WARN, _SRC_LOC, \
__FILE__ ":" func ": Unstable method!  Please report if it seems to " \
"work or not to the Shogun mailing list.  Thanking you in " \
"anticipation.  " __VA_ARGS__); }

#define SG_PRINT(...) { io->message(MSG_MESSAGEONLY, _SRC_LOC, __VA_ARGS__); }
#define SG_OBJ_PRINT(o, ...) { o->io->message(MSG_MESSAGEONLY, _SRC_LOC, __VA_ARGS__); }
#define SG_NOTIMPLEMENTED { io->not_implemented(_SRC_LOC); }
#define SG_GPL_ONLY { io->gpl_only(_SRC_LOC); }

#define SG_DONE() {								\
	if (SG_UNLIKELY(io->get_show_progress()))	\
		io->done();								\
}

// printf like function using the global SG_IO object
#define SG_IO env()->io()

#define SG_SGCDEBUG(...) {											\
	SG_IO->message(MSG_GCDEBUG, _SRC_LOC, __VA_ARGS__);\
}

#define SG_SDEBUG(...) {											\
	SG_IO->message(MSG_DEBUG, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_SINFO(...) {												\
	SG_IO->message(MSG_INFO, _SRC_LOC, __VA_ARGS__);	\
}

#define SG_SWARNING(...) { SG_IO->message(MSG_WARN, _SRC_LOC, __VA_ARGS__); }
#define SG_STHROW(Exception, ...)                                              \
	{                                                                          \
		SG_IO->template error<Exception>(                                      \
		    MSG_ERROR, _SRC_LOC, __VA_ARGS__);  \
	}
#define SG_SERROR(...) SG_STHROW(ShogunException, __VA_ARGS__)
#define SG_SPRINT(...) { SG_IO->message(MSG_MESSAGEONLY, _SRC_LOC, __VA_ARGS__); }

#define SG_SDONE() {								\
	if (SG_UNLIKELY(SG_IO->get_show_progress()))	\
		SG_IO->done();								\
}

#define SG_SNOTIMPLEMENTED { SG_IO->not_implemented(_SRC_LOC); }
#define SG_SGPL_ONLY { SG_IO->gpl_only(_SRC_LOC); }

#define ASSERT(x) {																	\
	if (SG_UNLIKELY(!(x)))																\
		SG_SERROR("assertion %s failed in %s file %s line %d\n",#x, __PRETTY_FUNCTION__, __FILE__, __LINE__)	\
}

#define REQUIRE(x, ...) {		\
	if (SG_UNLIKELY(!(x)))		\
		SG_SERROR(__VA_ARGS__)	\
}

#define REQUIRE_E(x, Exception, ...)                                           \
	{                                                                          \
		if (SG_UNLIKELY(!(x)))                                                 \
			SG_STHROW(Exception, __VA_ARGS__)                                  \
	}

/* help clang static analyzer to identify custom assertation functions */
#ifdef __clang_analyzer__
void _clang_fail(void) __attribute__((analyzer_noreturn));

#undef SG_ERROR(...)
#undef SG_SERROR(...)
#define SG_ERROR(...) _clang_fail();
#define SG_SERROR(...) _clang_fail();

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
		/** copy constructor */
		SGIO(const SGIO& orig);

		/** destructor */
		virtual ~SGIO();

		/** set loglevel
		 *
		 * @param level level of log messages
		 */
		void set_loglevel(EMessageType level);

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
		void message(EMessageType prio, const spdlog::source_loc& loc, const char* msg, const Args&... args) const;

		/** format and print a message
		 * @param prio message priority
		 * @param msg message format
		 * @param args arguments for formatting message
		 */
		template <typename... Args>
		void message(EMessageType prio, const char* msg, const Args&... args) const
		{
			message(prio, spdlog::source_loc{}, msg, args...);
		}

		/** format and print a message, and then throw an exception
		 * @tparam ExceptionType type of the exception to throw
		 * @param prio message priority
		 * @param loc source code location
		 * @param args arguments for formatting message
		 */
		template <typename Exception, typename... Args>
		void error(EMessageType prio, const spdlog::source_loc& loc, const char* msg, const Args&... args) const;

		/** format and print a message, and then throw an exception
		 * @tparam ExceptionType type of the exception to throw
		 * @param prio message priority
		 * @param loc source code location
		 * @param args arguments for formatting message
		 */
		template <typename Exception, typename... Args>
		void error(EMessageType prio, const char* msg, const Args&... args) const
		{
			error(prio, spdlog::source_loc{}, msg, args...);
		}

		/** print 'done' with priority INFO,
		 * but only if progress bar is enabled
		 *
		 */
		void done();

		/** print error message 'not implemented' */
		inline void not_implemented(const spdlog::source_loc& loc={}) const
		{
			error<ShogunException>(
			    MSG_ERROR, loc,
			    "Sorry, not yet implemented .\n");
		}

		/** print error message 'Only available with GPL parts.' */
		inline void gpl_only(const spdlog::source_loc& loc={}) const
		{
			error<ShogunException>(
			    MSG_ERROR, loc, "This feature is only "
			                                     "available if Shogun is built "
			                                     "with GPL codes.\n");
		}

		/** print warning message 'function deprecated' */
		inline void deprecated(const spdlog::source_loc& loc={}) const
		{
			message(MSG_WARN, loc,
					"This function is deprecated and will be removed soon.\n");
		}

		/** skip leading spaces
		 *
		 * @param str string in which to look for spaces
		 * @return string after after skipping leading spaces
		 */
		static char* skip_spaces(char* str);

		/** skip leading spaces + tabs
		 *
		 * @param str string in which to look for blanks
		 * @return string after after skipping leading blanks
		 */
		static char* skip_blanks(char* str);

		/** enable progress bar */
		inline void enable_progress()
		{
			show_progress=true;

			// static functions like CSVM::classify_example_helper call SG_PROGRESS
			if (SG_IO!=this)
				SG_IO->enable_progress();
		}

		/** disable progress bar */
		inline void disable_progress()
		{
			show_progress=false;

			// static functions like CSVM::classify_example_helper call SG_PROGRESS
			if (SG_IO!=this)
				SG_IO->disable_progress();
		}

		/** enable displaying of file and line when printing messages etc
		 *
		 * @param location location info (none, function, ...)
		 *
		 */
		void set_location_info(EMessageLocation location)
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

		/** increase reference counter
		 *
		 * @return reference count
		 */
		int32_t ref();

		/** display reference counter
		 *
		 * @return reference count
		 */
		int32_t ref_count() const;

		/** decrement reference counter and deallocate object if refcount is zero
		 * before or after decrementing it
		 *
		 * @return reference count
		 */
		int32_t unref();

		/** @return object name */
		inline const char* get_name() { return "SGIO"; }

	protected:
		void update_pattern();

	protected:
		/** if progress bar shall be shown */
		bool show_progress;
		/** if each print function should append filename and linenumber of
		 * where the print occurs etc */
		EMessageLocation location_info;
		/** whether syntax highlighting is enabled */
		bool syntax_highlight;

	private:
		std::shared_ptr<spdlog::logger> logger;
		RefCount* m_refcount;
};

namespace sgio_traits
{
	template <typename T>
	constexpr static inline auto cast_pointer_to_void(const T& t)
	{
		if constexpr (std::is_pointer<T>::value)
			return (void *) t;
		else if constexpr (std::is_array<T>::value)
			return &t[0];
		else
			return t;
	}
}

template <typename... Args>
void SGIO::message(EMessageType prio, const spdlog::source_loc& loc, const char* msg, const Args&... args) const
{
	// A solution to format using printf style
	// This is not optimal because it enforces formatting at call site
	// and the priority is checked twice
	const auto _prio = static_cast<spdlog::level::level_enum>(prio);
	if (logger->should_log(_prio))
	{
		std::string msg_formatted =
			fmt::sprintf(msg, sgio_traits::cast_pointer_to_void(args)...);
		logger->log(loc, _prio, msg_formatted);
	}
}

template <typename ExceptionType, typename... Args>
void SGIO::error(EMessageType prio, const spdlog::source_loc& loc, const char* msg, const Args&... args) const
{
	static_assert(std::is_nothrow_copy_constructible<ExceptionType>::value,
			  "ExceptionType must be nothrow copy constructible");
	message(prio, loc, msg, args...);
	throw ExceptionType(msg);
}
}
#endif // __SGIO_H__
