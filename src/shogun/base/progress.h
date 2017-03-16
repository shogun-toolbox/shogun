/*
* Written (W) 2017 Giovanni De Toni
*/

#ifndef __SG_PROGRESS_H__
#define __SG_PROGRESS_H__

#include <iterator>
#include <memory>
#include <string>

#include <shogun/base/init.h>
#include <shogun/base/range.h>
#include <shogun/io/SGIO.h>

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
		ProgressPrinter(const SGIO& io)
				: m_io(io), m_prefix("PROGRESS: "), m_mode(UTF8)
		{
		}
		ProgressPrinter(const SGIO& io, const std::string& prefix)
				: m_io(io), m_prefix(prefix), m_mode(UTF8)
		{
		}
		ProgressPrinter(const SGIO& io, const SG_PRG_MODE mode)
				: m_io(io), m_prefix("PROGRESS: "), m_mode(mode)
		{
		}
		ProgressPrinter(
				const SGIO& io, const std::string& prefix, const SG_PRG_MODE mode)
				: m_io(io), m_prefix(prefix), m_mode(mode)
		{
		}
		~ProgressPrinter()
		{
		}

		/** Print the progress bar */
		void print_progress() const
		{
			m_io.message(MSG_MESSAGEONLY, "", "", -1, "TEST\n");
		}

		/** Print the progress bar end */
		void print_end() const
		{
			m_io.message(MSG_MESSAGEONLY, "", "", -1, "100\n");
		}

	private:
		/** IO object */
		SGIO m_io;
		/** Prefix which will be printed before the progress bar */
		std::string m_prefix;
		/** Progres bar's char mode */
		SG_PRG_MODE m_mode;
		/** Maxmimum value */
		float64_t m_max_value;
		/** Minimum value */
		float64_t m_min_value;
		/** Last progress time */
		float64_t m_last_progress_time;
		/** Progress start time */
		float64_t m_progress_start_time;
		/** Last progress */
		float64_t m_last_progress;
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
			m_printer = std::make_shared<ProgressPrinter>(io);
		}
		PRange(Range<T> range, const SGIO& io, const SG_PRG_MODE mode)
				: m_range(range)
		{
			m_printer = std::make_shared<ProgressPrinter>(io, mode);
		}
		PRange(Range<T> range, const SGIO& io, const std::string prefix)
				: m_range(range)
		{
			m_printer = std::make_shared<ProgressPrinter>(io, prefix);
		}
		PRange(
				Range<T> range, const SGIO& io, const std::string prefix,
				const SG_PRG_MODE mode)
				: m_range(range)
		{
			m_printer = std::make_shared<ProgressPrinter>(io, prefix, mode);
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
				m_printer->print_progress();
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
			/* The ProgressPrinter object which will be used to show the progress bar*/
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

	private:
		/** Range we iterate over */
		Range<T> m_range;
		/** Observer that will print the actual progress bar */
		std::shared_ptr<ProgressPrinter> m_printer;
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
	inline PRange<T> progress(Range<T> range, const SGIO& io)
	{
		return PRange<T>(range, io);
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
	inline PRange<T> progress(Range<T> range)
	{
		return PRange<T>(range, *sg_io);
	}
};
#endif /* __SG_PROGRESS_H__ */