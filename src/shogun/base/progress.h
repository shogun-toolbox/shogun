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

#include <iterator>
#include <memory>
#include <string>

#include <shogun/base/init.h>
#include <shogun/base/range.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>

#if WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace shogun
{

	/** Possible print modes */
	enum SG_PRG_MODE
	{
		ASCII,
		UTF8
	};

	/** @class Printer that display the progress bar
	*
	*/
	class ProgressPrinter
	{
	public:
		/** Creates a ProgressPrinter instance given an SGIO object
		*
		* @param    io  SGIO object
		*/
		ProgressPrinter(
		    const SGIO& io, float64_t max_value, float64_t min_value)
		    : m_io(io), m_max_value(max_value), m_min_value(min_value),
		      m_prefix("PROGRESS: "), m_mode(UTF8), m_last_progress(0),
		      m_last_progress_time(0), m_progress_start_time(0)
		{
		}
		ProgressPrinter(
		    const SGIO& io, float64_t max_value, float64_t min_value,
		    const std::string& prefix)
		    : m_io(io), m_max_value(max_value), m_min_value(min_value),
		      m_prefix(prefix), m_mode(UTF8), m_last_progress(0),
		      m_last_progress_time(0), m_progress_start_time(0)
		{
		}
		ProgressPrinter(
		    const SGIO& io, float64_t max_value, float64_t min_value,
		    const SG_PRG_MODE mode)
		    : m_io(io), m_max_value(max_value), m_min_value(min_value),
		      m_prefix("PROGRESS: "), m_mode(mode), m_last_progress(0),
		      m_last_progress_time(0), m_progress_start_time(0)
		{
		}
		ProgressPrinter(
		    const SGIO& io, float64_t max_value, float64_t min_value,
		    const std::string& prefix, const SG_PRG_MODE mode)
		    : m_io(io), m_max_value(max_value), m_min_value(min_value),
		      m_prefix(prefix), m_mode(mode), m_last_progress(0),
		      m_last_progress_time(0), m_progress_start_time(0)
		{
		}
		~ProgressPrinter()
		{
		}

		/** Print the progress bar */
		void print_progress(float64_t current_value) const
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
			float64_t runtime = CTime::get_curtime();

			if (difference > 0.0)
				v = 100 * (current_value - m_min_value + 1) /
				    (m_max_value - m_min_value + 1);

			// Set up chunk size
			size_chunk = difference / (float64_t)progress_bar_space;

			if (m_last_progress == 0)
			{
				m_last_progress_time = runtime;
				m_progress_start_time = runtime;
				m_last_progress = v;
			}
			else
			{
				m_last_progress = v - 1e-6;

				if ((v != 100.0) && (runtime - m_last_progress_time < 0.5))
				{
					// This is made to display correctly the percentage
					// if the algorithm execution is too fast
					if (current_value >= m_max_value - 1)
					{
						v = 100;
						m_last_progress = v - 1e-6;
						snprintf(
						    str, sizeof(str), "%%s %.2f    %%1.1f "
						                      "seconds remaining    %%1.1f "
						                      "seconds total\r");
						m_io.message(
						    MSG_MESSAGEONLY, "", "", -1, str, m_prefix.c_str(),
						    v, estimate, total_estimate);
					}
					return;
				}

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
				if (current_value > i * size_chunk)
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

			// If we arrive to the end, we print a new line (fancier output)
			if (current_value >= m_max_value)
				m_io.message(MSG_MESSAGEONLY, "", "", -1, "\n");
		}

		/** Print the progress bar end */
		void print_end() const
		{
			// Check if the progress was enabled
			if (!m_io.get_show_progress())
				return;

			print_progress(m_max_value);
		}

		/** @return last progress as a percentage */
		inline float64_t get_last_progress() const
		{
			return m_last_progress;
		}

	private:
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

		void set_screen_size() const
		{
#if WIN32
			CONSOLE_SCREEN_BUFFER_INFO csbi;
			GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
			m_columns_num = csbi.srWindow.Right - csbi.srWindow.Left + 1;
			m_rows_num = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#else
			struct winsize wind;
			ioctl(STDOUT_FILENO, TIOCGWINSZ, &wind);
			m_columns_num = wind.ws_col;
			m_rows_num = wind.ws_row;
#endif
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
		/** Create a progress range given an actual range and an io manager
		*
		* @param range  range object
		* @param io     io manager
		*/
		PRange(Range<T> range, const SGIO& io) : m_range(range)
		{
			set_up_range();
			m_printer =
			    std::make_shared<ProgressPrinter>(io, end_range, begin_range);
		}
		PRange(Range<T> range, const SGIO& io, const SG_PRG_MODE mode)
		    : m_range(range)
		{
			set_up_range();
			m_printer = std::make_shared<ProgressPrinter>(
			    io, end_range, begin_range, mode);
		}
		PRange(Range<T> range, const SGIO& io, const std::string prefix)
		    : m_range(range)
		{
			set_up_range();
			m_printer = std::make_shared<ProgressPrinter>(
			    io, end_range, begin_range, prefix);
		}
		PRange(
		    Range<T> range, const SGIO& io, const std::string prefix,
		    const SG_PRG_MODE mode)
		    : m_range(range)
		{
			set_up_range();
			m_printer = std::make_shared<ProgressPrinter>(
			    io, end_range, begin_range, prefix, mode);
		}

		/** @class Wrapper for Range<T>::Iterator spawned by @ref PRange. */
		class PIterator : public std::iterator<std::input_iterator_tag, T>
		{
		public:
			PIterator(
			    typename Range<T>::Iterator value,
			    std::shared_ptr<ProgressPrinter> shrd_ptr)
			    : m_value(value), m_printer(shrd_ptr)
			{
			}
			PIterator(const PIterator& other)
			    : m_value(other.m_value), m_printer(other.m_printer)
			{
			}
			PIterator(PIterator&& other)
			    : m_value(other.m_value), m_printer(other.m_printer)
			{
			}
			PIterator& operator=(const PIterator&) = delete;
			PIterator& operator++()
			{
				// Every time we update the iterator we print
				// also the updated progress bar
				m_printer->print_progress((*m_value));
				m_value++;
				return *this;
			}
			PIterator& operator++(int)
			{
				PIterator tmp(*this, this->m_printer);
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
					m_printer->print_end();
				return this->m_value != other.m_value;
			}
			bool operator==(const PIterator& other)
			{
				return this->m_value == other.m_value;
			}

		private:
			/* The ProgressPrinter object which will be used to show the
			 * progress bar*/
			std::shared_ptr<ProgressPrinter> m_printer;
			/* The wrapped range */
			typename Range<T>::Iterator m_value;
		};

		/** Create the iterator that corresponds to the start of the range*/
		PIterator begin() const
		{
			return PIterator(m_range.begin(), m_printer);
		}

		/** Create the iterator that corresponds to the end of the iterator*/
		PIterator end() const
		{
			return PIterator(m_range.end(), m_printer);
		}

		/** @return last progress as a percentage */
		inline float64_t get_last_progress() const
		{
			return m_printer->get_last_progress();
		}

	private:
		void set_up_range()
		{
			begin_range = *(m_range.begin());
			end_range = *(m_range.end());
		}

		/** Range we iterate over */
		Range<T> m_range;
		/** Observer that will print the actual progress bar */
		std::shared_ptr<ProgressPrinter> m_printer;
		float64_t begin_range;
		float64_t end_range;
	};

	/** Creates @ref PRange given a range.
	 *
	 * @code
	 *  for (auto i : progress(range(0, 100), io)) { ... }
	 * @endcode
	 *
	 * @param   range   range used
	 * @param   io      SGIO object
	 */
	template <typename T>
	inline PRange<T>
	progress(Range<T> range, const SGIO& io, SG_PRG_MODE mode = UTF8)
	{
		return PRange<T>(range, io, mode);
	}

	/** Creates @ref PRange given a range that uses the global SGIO
	 *
	 * @code
	 *  for (auto i : progress( range(0, 100) ) ) { ... }
	 * @endcode
	 *
	 * @param   range   range used
	 */
	template <typename T>
	inline PRange<T> progress(Range<T> range, SG_PRG_MODE mode = UTF8)
	{
		return PRange<T>(range, *sg_io, mode);
	}
};
#endif /* __SG_PROGRESS_H__ */