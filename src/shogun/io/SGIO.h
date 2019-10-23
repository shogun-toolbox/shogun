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

#include <shogun/base/ShogunEnv.h>
#include <shogun/io/fmt/fmt.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>

#include <locale.h>
#include <string.h>
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
} // namespace spdlog

namespace shogun
{

#ifdef DARWIN
#include <Availability.h>
#endif // DARWIN

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

#define SOURCE_LOCATION                                                        \
	io::SourceLocation{__FILE__, __LINE__, __PRETTY_FUNCTION__}

#ifdef DEBUG_BUILD
#define SG_TRACE(...)                                                          \
	env()->io()->message(io::MSG_TRACE, SOURCE_LOCATION, __VA_ARGS__)

#define SG_DEBUG(...)                                                          \
	env()->io()->message(io::MSG_DEBUG, SOURCE_LOCATION, __VA_ARGS__);

#define ASSERT(x)                                                              \
	{                                                                          \
		if (SG_UNLIKELY(!(x)))                                                 \
			error(SOURCE_LOCATION, "assertion {} failed", #x);                 \
	}
#else
#define SG_TRACE(...) (void)0
#define SG_DEBUG(...) (void)0;
#define ASSERT(...) (void)0;
#endif // DEBUG_BUILD

#ifdef __clang_analyzer__
	void _clang_fail(void) __attribute__((analyzer_noreturn));
#endif

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
		char* start;
		/** end */
		char* end;
	};

	namespace io
	{
		/** The io libs output [DEBUG] etc in front of every message 'higher'
		 * messages filter output depending on the loglevel, i.e. WARN messages
		 * will print all MSG_WARN TO MSG_MESSAGEONLY messages.
		 */
		enum EMessageType
		{
			MSG_TRACE = 0,
			MSG_DEBUG = 1,
			MSG_INFO = 2,
			MSG_WARN = 3,
			MSG_ERROR = 4,
			MSG_CRITICAL = 5,
			MSG_MESSAGEONLY = 6
		};

#ifndef SWIG
		struct SourceLocation
		{
			constexpr SourceLocation(
			    const char* file_ = "", int32_t line_ = 0,
			    const char* function_ = "")
			    : file(file_), line(line_), function(function_)
			{
			}

			const char* file;
			int32_t line;
			const char* function;
		};
#endif

		/** @brief Class SGIO, used to do input output operations throughout
		 * shogun.
		 *
		 * Any debug or error or progress message is passed through the
		 * functions of this class to be in the end written to the screen. Note
		 * that messages don't have to be written to stdout or stderr, but can
		 * be redirected to a file.
		 */
		class SGIO
		{
		public:
			/** default constructor */
			SGIO();

			/** destructor */
			virtual ~SGIO();

			/** redirects stdout to another sink */
			void redirect_stdout(const std::shared_ptr<spdlog::sinks::sink>& sink);

			/** redirects stderr to another sink */
			void redirect_stderr(const std::shared_ptr<spdlog::sinks::sink>& sink);

			/** (re)initializes the default (asynchronous) logger
			 *
			 * @param queue_size size of the message queue
			 * @param n_threads number of logging threads
			 */
			void init_default_logger(
			    uint64_t queue_size = 128, uint64_t n_threads = 1);

			/** (re)initializes the default sink
			 */
			void init_default_sink();

			/** get loglevel
			 *
			 * @return level of log messages
			 */
			EMessageType get_loglevel() const;

			/** set loglevel
			 *
			 * @param level level of log messages
			 */
			void set_loglevel(EMessageType level);

			/** @return whether loglevel is above specified level and thus the
			 * message should be printed
			 */
			bool should_log(EMessageType prio) const;

			/** get show_progress
			 *
			 * @return if progress bar is shown
			 */
			inline bool get_show_progress() const
			{
				return show_progress;
			}

			/** get syntax highlight
			 *
			 * @return if syntax highlighting is enabled
			 */
			inline bool get_syntax_highlight() const
			{
				return syntax_highlight;
			}

			/** enable syntax highlighting */
			void enable_syntax_highlighting();

			/** disable syntax highlighting */
			void disable_syntax_highlighting();

			/** enable progress bar */
			inline void enable_progress()
			{
				show_progress = true;
			}

			/** disable progress bar */
			inline void disable_progress()
			{
				show_progress = false;
			}

#ifndef SWIG
			/** format and print a message
			 * @param prio message priority
			 * @param loc source code location
			 * @param msg message format
			 * @param args arguments for formatting message
			 */
			template <typename... Args>
			void message(
			    EMessageType prio, const SourceLocation& loc,
			    const char* format, const Args&... args) const;

			/**
			 * Return a C string from the substring
			 * @param s substring
			 * @return new C string representation
			 */
			static char* c_string_of_substring(substring s);

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
#endif // SWIG

		private:
			/** Prints a formatted message */
			void message_(
			    EMessageType prio, const SourceLocation& loc,
			    const fmt::string_view& msg) const;

			/** if progress bar shall be shown */
			bool show_progress;

			/** whether syntax highlighting is enabled */
			bool syntax_highlight;

			class RedirectSink;
			std::shared_ptr<RedirectSink> io_sink;
			std::shared_ptr<spdlog::logger> io_logger;
			std::shared_ptr<spdlog::details::thread_pool> thread_pool;
		};

#ifndef SWIG
		template <typename... Args>
		void SGIO::message(
		    EMessageType prio, const SourceLocation& loc, const char* format,
		    const Args&... args) const
		{
			if (should_log(prio))
			{
				fmt::memory_buffer msg;
				fmt::format_to(msg, format, args...);
				message_(prio, loc, fmt::string_view(msg.data(), msg.size()));
			}
		}

		template <typename... Args>
		static inline void print(const char* format, const Args&... args)
		{
			env()->io()->message(MSG_MESSAGEONLY, {}, format, args...);
		}

		template <typename... Args>
		static inline void info(const char* format, const Args&... args)
		{
			env()->io()->message(MSG_INFO, {}, format, args...);
		}

		template <typename... Args>
		static inline void warn(const char* format, const Args&... args)
		{
			env()->io()->message(MSG_WARN, {}, format, args...);
		}

		static inline void progress_done()
		{
			if (SG_UNLIKELY(env()->io()->get_show_progress()))
			{
				info("done.");
			}
		}
#endif // SWIG
	}  // namespace io

#ifndef SWIG
	template <typename ExceptionType = ShogunException, typename... Args>
	static inline void error(
	    const io::SourceLocation& loc, const char* format, const Args&... args)
	{
		// help clang static analyzer to identify custom assertation functions
#ifdef __clang_analyzer__
		_clang_fail();
#else
		static_assert(
		    std::is_nothrow_copy_constructible<ExceptionType>::value,
		    "ExceptionType must be nothrow copy constructible");

		fmt::memory_buffer msg;
		fmt::format_to(msg, format, args...);
		msg.push_back('\0');
		env()->io()->message(io::MSG_ERROR, loc, msg.data());
		throw ExceptionType(msg.data());
#endif /* __clang_analyzer__ */
	}

	template <typename ExceptionType = ShogunException, typename... Args>
	static inline void error(const char* format, const Args&... args)
	{
		error<ExceptionType>(io::SourceLocation{}, format, args...);
	}

	template <
	    typename ExceptionType = ShogunException, typename Condition,
	    typename... Args>
	static inline void
	require(const Condition& condition, const char* format, const Args&... args)
	{
		if (SG_UNLIKELY(!condition))
		{
			error<ExceptionType>(format, args...);
		}
	}

	/** print error message 'not implemented' */
	static inline void not_implemented(const io::SourceLocation& loc = {})
	{
		error<ShogunException>(loc, "Sorry, not yet implemented.");
	}

	/** print error message 'Only available with GPL parts.' */
	static inline void gpl_only(const io::SourceLocation& loc = {})
	{
		error<ShogunException>(
		    loc, "This feature is only "
		         "available if Shogun is built "
		         "with GPL codes.");
	}


	static inline void unstable(const io::SourceLocation& loc = {})
	{
		env()->io()->message(
		    io::MSG_WARN, loc,
		    "Unstable method!  Please report if it seems to "
		    "work or not to the Shogun mailing list.  Thanking you in "
		    "anticipation.");
	}
#endif // SWIG
} // namespace shogun
#endif // __SGIO_H__
