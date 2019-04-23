/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
*/

#ifndef __SG_PROGRESS_H__
#define __SG_PROGRESS_H__

#include <functional>
#include <iterator>
#include <memory>
#include <string>

#include <shogun/base/init.h>
#include <shogun/base/range.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Lock.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>

#ifdef WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>

#endif

namespace shogun
{
#define SG_PROGRESS(...)                                                       \
	progress(                                                                  \
	    std::string(this->get_name()) + "::" + std::string(__FUNCTION__),      \
	    *this->io, __VA_ARGS__)

#define SG_SPROGRESS(...) progress(__FUNCTION__, __VA_ARGS__)

	/** Possible print modes */
	enum SG_PRG_MODE
	{
		ASCII,
		UTF8
	};

	/**
	 * @class Printer class that displays the progress bar.
	 */
	class ProgressPrinter
	{
	public:
		/**
		 * Creates a @ref ProgressPrinter instance.
		 * @param io SGIO object which will be used to print the progress bar.
		 * @param max_value interval maximum value.
		 * @param min_value interval minimum value.
		 * @param prefix string which will be printed before the progress bar.
		 * @param mode char mode (UTF8, ASCII etc.).
		 */
		ProgressPrinter(
		    const SGIO& io, float64_t max_value, float64_t min_value,
		    const std::string& prefix, const SG_PRG_MODE mode)
		    : m_io(io), m_max_value(max_value), m_min_value(min_value),
		      m_prefix(prefix), m_mode(mode), m_columns_num(0), m_rows_num(0),
		      m_last_progress(0), m_last_progress_time(0),
		      m_progress_start_time(Time::get_curtime()),
		      m_current_value(min_value)
		{
		}
		~ProgressPrinter()
		{
		}

		/**
		 * Increment and print the progress bar.
		 * Everything is locked to prevent race conditions
		 * or characters overlapping (especially within
		 * multi threaded environments).
		 */
		void print_progress() const
		{
			lock.lock();
			if (m_current_value.load() - m_min_value >
			    m_max_value - m_min_value)
			{
				increment();
				lock.unlock();
				return;
			}
			print_progress_impl();
			if (m_current_value.load() - m_min_value ==
			    m_max_value - m_min_value)
			{
				print_end();
				increment();
				lock.unlock();
				return;
			}
			increment();
			lock.unlock();
		}

		void print_progress_absolute(
		    float64_t current_val, float64_t val, float64_t min_val,
		    float64_t max_val)
		{
			lock.lock();
			if (val - m_min_value > m_max_value - m_min_value)
			{
				lock.unlock();
				return;
			}
			print_progress_absolute_impl(current_val, val, min_val, max_val);
			if (val - m_min_value == m_max_value - m_min_value)
			{
				print_end();
				lock.unlock();
				return;
			}
			lock.unlock();
		}

		/**
		 * Manually increment to max size the current value
		 * to print a complete progress bar.
		 */
		void premature_end()
		{
			if (m_current_value.load() < m_max_value - 1)
				m_current_value.store(m_max_value);
		}

		/** @return last progress as a percentage. */
		inline float64_t get_current_progress() const
		{
			return m_current_value.load();
		}

	private:
		/**
		 * Logic implementation of the progress bar.
		 */
		void print_progress_impl() const
		{

			// Check if the progress was enabled
			if (!m_io.get_show_progress())
				return;

			if (m_max_value <= m_min_value)
				return;

			// Check for terminal dimension. This is for provide
			// a minimal resize functionality.
			set_screen_size();

			float64_t difference = m_max_value - m_min_value, v = -1,
			          estimate = 0, total_estimate = 0;
			float64_t size_chunk = -1;

			// Check if we have enough space to show the progress bar
			// Use only a fraction of it to account for the size of the
			// time displayed (decimals and integer).
			int32_t progress_bar_space =
			    (m_columns_num - 50 - m_prefix.length()) * 0.9;

			// TODO: this guy here brokes testing
			// REQUIRE(
			//    progress_bar_space > 0,
			//    "Not enough terminal space to show the progress bar!\n")

			char str[1000];
			float64_t runtime = Time::get_curtime();

			if (difference > 0.0)
				v = 100 * (m_current_value.load() - m_min_value) /
				    (m_max_value - m_min_value);

			// Set up chunk size
			size_chunk = difference / (float64_t)progress_bar_space;

			if (m_last_progress == 0)
			{
				m_last_progress_time = runtime;
				m_last_progress = v;
			}
			else
			{
				m_last_progress = v - 1e-6;

				if ((v != 100.0) && (runtime - m_last_progress_time < 0.5))
					return;

				m_last_progress_time = runtime;
				estimate = (1 - v / 100) *
				           (m_last_progress_time - m_progress_start_time) /
				           (v / 100);
				total_estimate =
				    (m_last_progress_time - m_progress_start_time) / (v / 100);
			}

			/** Print the actual progress bar to screen **/
			m_io.message(MSG_MESSAGEONLY, "", "", -1, "%s |", m_prefix.c_str());
			for (index_t i = 1; i < progress_bar_space; i++)
			{
				if (m_current_value.load() - m_min_value > i * size_chunk)
				{
					m_io.message(
					    MSG_MESSAGEONLY, "", "", -1, "%s",
					    get_pb_char().c_str());
				}
				else
				{
					m_io.message(MSG_MESSAGEONLY, "", "", -1, " ");
				}
			}
			m_io.message(MSG_MESSAGEONLY, "", "", -1, "| %.2f\%", v);

			if (estimate > 120)
			{
				snprintf(
				    str, sizeof(str),
				    "   %%1.1f minutes remaining  %%1.1f minutes total\r");
				m_io.message(
				    MSG_MESSAGEONLY, "", "", -1, str, estimate / 60,
				    total_estimate / 60);
			}
			else
			{
				snprintf(
				    str, sizeof(str),
				    "   %%1.1f seconds remaining  %%1.1f seconds total\r");
				m_io.message(
				    MSG_MESSAGEONLY, "", "", -1, str, estimate, total_estimate);
			}
		}

		/**
		 * Logic implementation fo the absolute progress bar.
		 */
		void print_progress_absolute_impl(
		    float64_t current_val, float64_t val, float64_t min_value,
		    float64_t max_value) const
		{
			// Check if the progress was enabled
			if (!m_io.get_show_progress())
				return;

			m_current_value.store(current_val);

			if (max_value <= min_value)
				return;

			// Check for terminal dimension. This is for provide
			// a minimal resize functionality.
			set_screen_size();

			float64_t difference = max_value - min_value, v = -1, estimate = 0,
			          total_estimate = 0;
			float64_t size_chunk = -1;

			// Check if we have enough space to show the progress bar
			// Use only a fraction of it to account for the size of the
			// time displayed (decimals and integer).
			int32_t progress_bar_space =
			    (m_columns_num - 50 - m_prefix.length()) * 0.9;

			// TODO: this guy here brokes testing
			// REQUIRE(
			//    progress_bar_space > 0,
			//    "Not enough terminal space to show the progress bar!\n")

			char str[1000];
			float64_t runtime = Time::get_curtime();

			if (difference > 0.0)
				v = 100 * (val - min_value) / (max_value - min_value);

			// Set up chunk size
			size_chunk = difference / (float64_t)progress_bar_space;

			if (m_last_progress == 0)
			{
				m_last_progress_time = runtime;
				m_last_progress = v;
			}
			else
			{
				m_last_progress = v - 1e-6;

				if ((v != 100.0) && (runtime - m_last_progress_time < 0.5))
					return;

				m_last_progress_time = runtime;
				estimate = (1 - v / 100) *
				           (m_last_progress_time - m_progress_start_time) /
				           (v / 100);
				total_estimate =
				    (m_last_progress_time - m_progress_start_time) / (v / 100);
			}

			/** Print the actual progress bar to screen **/
			m_io.message(MSG_MESSAGEONLY, "", "", -1, "%s |", m_prefix.c_str());
			for (index_t i = 1; i < progress_bar_space; i++)
			{
				if (m_current_value.load() - min_value > i * size_chunk)
				{
					m_io.message(
					    MSG_MESSAGEONLY, "", "", -1, "%s",
					    get_pb_char().c_str());
				}
				else
				{
					m_io.message(MSG_MESSAGEONLY, "", "", -1, " ");
				}
			}
			m_io.message(MSG_MESSAGEONLY, "", "", -1, "| %.2f\%", current_val);

			if (estimate > 120)
			{
				snprintf(
				    str, sizeof(str),
				    "   %%1.1f minutes remaining  %%1.1f minutes total\r");
				m_io.message(
				    MSG_MESSAGEONLY, "", "", -1, str, estimate / 60,
				    total_estimate / 60);
			}
			else
			{
				snprintf(
				    str, sizeof(str),
				    "   %%1.1f seconds remaining  %%1.1f seconds total\r");
				m_io.message(
				    MSG_MESSAGEONLY, "", "", -1, str, estimate, total_estimate);
			}
		}

		/** Print the progress bar end. */
		void print_end() const
		{
			// Check if the progress was enabled
			if (!m_io.get_show_progress())
				return;

			m_io.message(MSG_MESSAGEONLY, "", "", -1, "\n");
		}

		/**
		 * Return the char which will be used to print the progress.
		 * @return UTF8/ASCII string
		 */
		std::string get_pb_char() const
		{
			switch (m_mode)
			{
			case ASCII:
				return m_ascii_char;
			case UTF8:
				return m_utf8_char;
			default:
				return m_ascii_char;
			}
		}

		/**
		 * Get the terminal's screen size (Windows and Unix).
		 */
		void set_screen_size() const
		{
#ifdef WIN32
			CONSOLE_SCREEN_BUFFER_INFO csbi;
			GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
			m_columns_num = csbi.srWindow.Right - csbi.srWindow.Left + 1;
			m_rows_num = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#else
			struct winsize wind;
			wind.ws_col = 0;
			wind.ws_row = 0;
			ioctl(STDOUT_FILENO, TIOCGWINSZ, &wind);
			m_columns_num = wind.ws_col;
			m_rows_num = wind.ws_row;
#endif
		}

		/* Increment the current value (atomically) */
		void increment() const
		{
			m_current_value++;
		}

		/** IO object */
		SGIO m_io;
		/** Maxmimum value */
		float64_t m_max_value;
		/** Minimum value */
		float64_t m_min_value;
		/** Prefix which will be printed before the progress bar */
		std::string m_prefix;
		/** Progres bar's char mode */
		SG_PRG_MODE m_mode;
		/** ASCII char */
		std::string m_ascii_char = "#";
		/** UTF8 char */
		std::string m_utf8_char = "\u2588";
		/* Screen column number*/
		mutable int32_t m_columns_num;
		/* Screen row number*/
		mutable int32_t m_rows_num;
		/** Last progress */
		mutable float64_t m_last_progress;
		/** Last progress time */
		mutable float64_t m_last_progress_time;
		/** Progress start time */
		mutable float64_t m_progress_start_time;
		/** Current value */
		mutable std::atomic<int64_t> m_current_value;
		/** Lock for multithreaded operations **/
		mutable CLock lock;
	};

	/** @class Helper class to show a progress bar given a range.
	 *
	 * @code
	 *  for (auto i : PRange<int>(Range<int>(1, 10), io)) { ... }
	 * @endcode
	 */
	template <typename T>
	class PRange
	{
	public:
		/**
		 * Constructor, initialize the progress bar manager.
		 *
		 * @param range the range to loop over
		 * @param io the SGIO object which will be used to print the progress
		 * bar
		 * @param prefix the string prefix which will be printed before the
		 * progress bar
		 * @param mode the char mode used to print the progress bar (ASCII, UTF8
		 * etc.)
		 * @param condition premature stop condition for the loop
		 */
		PRange(
		    Range<T> range, const SGIO& io, const std::string prefix,
		    const SG_PRG_MODE mode, std::function<bool()> condition)
		    : m_range(range), m_condition(condition)
		{
			set_up_range();
			m_printer = std::make_shared<ProgressPrinter>(
			    io, m_end_range, m_begin_range, prefix, mode);
		}

		/** @class Wrapper for Range<T>::Iterator spawned by @ref PRange. */
		class PIterator : public std::iterator<std::input_iterator_tag, T>
		{
		public:
			/**
			 * Initialize the PIterator object.
			 * @param value the @ref Range<T>:Iterator object.
			 * @param shrd_ptr the @ref ProgressPrinter object.
			 * @param condition premature stop condition for the loop.
			 */
			PIterator(
			    typename Range<T>::Iterator value,
			    std::shared_ptr<ProgressPrinter> shrd_ptr,
			    std::function<bool()> condition)
			    : m_value(value), m_printer(shrd_ptr), m_condition(condition)
			{
			}
			PIterator(const PIterator& other)
			    : m_value(other.m_value), m_printer(other.m_printer),
			      m_condition(other.m_condition)
			{
			}
			PIterator(PIterator&& other)
			    : m_value(other.m_value), m_printer(other.m_printer),
			      m_condition(other.m_condition)
			{
			}
			PIterator& operator=(const PIterator&) = delete;
			PIterator& operator++()
			{
				// Every time we update the iterator we print
				// also the updated progress bar
				m_printer->print_progress();
				m_value++;
				return *this;
			}
			PIterator operator++(int)
			{
				PIterator tmp(*this);
				++*this;
				return tmp;
			}
			T operator*()
			{
				// Since PIterator is a wrapper we have
				// to return the actual value of the
				// wrapped iterator
				return *m_value;
			}
			bool operator!=(const PIterator& other)
			{
				if (!(this->m_value != other.m_value))
				{
					m_printer->premature_end();
					m_printer->print_progress();
					return false;
				}
				bool result = evaluate_condition();
				return (this->m_value != other.m_value) && result;
			}

		private:
			/**
			 * Evaluate the premature stop condition.
			 * @return return value of the condition.
			 */
			bool evaluate_condition()
			{
				if (!m_condition())
				{
					m_printer->premature_end();
					m_printer->print_progress();
				}
				return m_condition();
			}

			/* The wrapped range */
			typename Range<T>::Iterator m_value;
			/* The ProgressPrinter object which will be used to show the
			 * progress bar*/
			std::shared_ptr<ProgressPrinter> m_printer;
			/* The function which will contain the custom condition
			 * to premature stop the loop */
			std::function<bool()> m_condition;
		};

		/** Create the iterator that corresponds to the start of the range.
		 *  Used within the range-based loop version of the progress bar.
		 *
		 * @code
		 * 	for (auto i: progress(range(0, 10), io, ASCII))
		 * 	{
		 * 		//Do stuff
		 * 	}
		 * @endcode
		 *
		 * @return @ref PIterator that represents the start of the range
		 */
		PIterator begin() const
		{
			return PIterator(m_range.begin(), m_printer, m_condition);
		}

		/** Create the iterator that corresponds to the end of the range.
		 * Used within the range-based loop version of the progress bar.
		 *
		 * @code
		 * 	for (auto i: progress(range(0, 10), io, ASCII))
		 * 	{
		 * 		//Do stuff
		 * 	}
		 * @endcode
		 *
		 * @return @ref PIterator that represent the end of the range.
		 */
		PIterator end() const
		{
			return PIterator(m_range.end(), m_printer, m_condition);
		}

		/**
		 * Return the current progress bar value.
		 * Used for testing purposes.
		 * @return current progress bar value.
		 */
		inline float64_t get_current_progress() const
		{
			return m_printer->get_current_progress();
		}

		/**
		 * Print the progress bar. This method must be called
		 * each time we want the progress bar to be updated.
		 * @code
		 * 	auto pr = progress(range(0,10), ASCII);
		 * 	for (int i=0; i<10; i++)
		 * 	{
		 * 		// Do stuff
		 * 		pr.print_progress();
		 * 	}
		 * 	pr.complete();
		 * @endcode
		 */
		void print_progress() const
		{
			m_printer->print_progress();
		}

		/**
		 * Print the absolute progress bar. This method must be called
		 * each time we want the progress bar to be updated.
		 *
		 * @param current_val current value
		 * @param val value
		 * @param min_val minimum value
		 * @param max_val maximum value
		 */
		void print_absolute(
		    float64_t current_val, float64_t val, float64_t min_value,
		    float64_t max_value) const
		{
			m_printer->print_progress_absolute(
			    current_val, val, min_value, max_value);
		}

		/**
		 * Print the progress bar end. This method must be called
		 * one time, after the loop.
		 * @code
		 * 	auto pr = progress(range(0,10), ASCII);
		 * 	for (int i=0; i<10; i++)
		 * 	{
		 * 		// Do stuff
		 * 		pr.print_progress();
		 * 	}
		 * 	pr.complete();
		 * @endcode
		 */
		void complete() const
		{
			m_printer->premature_end();
			m_printer->print_progress();
		}

		/**
		 * Print the progress bar end. This method must be called
		 * one time, after the loop.
		 * @code
		 * 	auto pr = progress(range(0,10), ASCII);
		 * 	for (int i=0; i<10; i++)
		 * 	{
		 * 		// Do stuff
		 * 		pr.print_absolute();
		 * 	}
		 * 	pr.complete_absolute();
		 * @endcode
		 */
		void complete_absolute() const
		{
			m_printer->print_progress_absolute(100, 100, 0, 100);
		}

	private:
		/**
		 * Set up progress range.
		 */
		void set_up_range()
		{
			m_begin_range = *(m_range.begin());
			m_end_range = *(m_range.end());
		}

		/** Range we iterate over */
		Range<T> m_range;
		/** Observer that will print the actual progress bar */
		std::shared_ptr<ProgressPrinter> m_printer;
		/* Start of the range */
		float64_t m_begin_range;
		/* End of the range */
		float64_t m_end_range;
		/* Function which store the premature stop condition */
		std::function<bool()> m_condition = []() { return true; };
	};

	/** Creates @ref PRange given a range.
	 *
	 * @code
	 *  for (auto i : progress(range(0, 100), io)) { ... }
	 * @endcode
	 *
	 * @param   range   range used
	 * @param   io      SGIO object
	 * @param	mode	char printing mode (default: UTF8)
	 * @param	prefix  string which will be printed before the progress bar
	 * (default: PROGRESS: )
	 * @param	condition	premature stopping condition
	 */
	template <typename T>
	inline PRange<T> progress(
	    std::string prefix, const SGIO& io, Range<T> range,
	    SG_PRG_MODE mode = UTF8,
	    std::function<bool()> condition = []() { return true; })
	{
		return PRange<T>(range, io, prefix, mode, condition);
	}

	/** Creates @ref PRange given a range that uses the global SGIO
	 *
	 * @code
	 *  for (auto i : progress( range(0, 100) ) ) { ... }
	 * @endcode
	 *
	 * @param   range   range used
	 * @param	mode	char printing mode (default: UTF8)
	 * @param	prefix  string which will be printed before the progress bar
	 * (default: PROGRESS: )
	 * @param	condition	premature stopping condition
	 */
	template <typename T>
	inline PRange<T> progress(
	    std::string prefix, Range<T> range, SG_PRG_MODE mode = UTF8,
	    std::function<bool()> condition = []() { return true; })
	{
		return PRange<T>(range, *sg_io, prefix, mode, condition);
	}

	/** Creates @ref PRange given a range that uses the global SGIO
	 *
	 * @param range range used
	 * @param condition premature stopping condition
	 */
	template <typename T>
	inline PRange<T> progress(
	    std::string prefix, Range<T> range, std::function<bool()> condition)
	{
		return PRange<T>(range, *sg_io, prefix, UTF8, condition);
	}
	/** Creates @ref PRange given a range and a stopping condition
	 *
	 * @param range range used
	 * @param io SGIO object
	 * @param condition premature stopping condition
	 */
	template <typename T>
	inline PRange<T> progress(
	    std::string prefix, const SGIO& io, Range<T> range,
	    std::function<bool()> condition)
	{
		return PRange<T>(range, io, prefix, UTF8, condition);
	}
};
#endif /* __SG_PROGRESS_H__ */
