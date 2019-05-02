/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ARFFFILE_H
#define SHOGUN_ARFFFILE_H

#include <shogun/base/init.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <cctype>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace shogun
{
	namespace arff_detail
	{
		/**
		 * Checks if string is blank
		 * @param line to check
		 * @return bool whether line is empty
		 */
		SG_FORCED_INLINE bool string_is_blank(const std::string& line)
		{
			return line.find_first_not_of(" \t\r\f\v") == std::string::npos;
		}

		/**
		 * Splits a line given a set of delimiter characters
		 *
		 * @tparam Out type of container
		 * @param s string to split
		 * @param delimiters a set of delimiter character
		 * @param result dynamic container where tokens are stored
		 */
		template <typename Out>
		void split(const std::string& s, const char* delimiters, Out result)
		{
			std::stringstream ss(s);
			std::string line;
			while (std::getline(ss, line))
			{
				size_t prev = 0, pos;
				while ((pos = line.find_first_of(delimiters, prev)) !=
				       std::string::npos)
				{
					if (pos > prev)
						*(result++) = line.substr(prev, pos - prev);
					prev = pos + 1;
				}
				if (prev < line.length())
					*(result++) = line.substr(prev, std::string::npos);
			}
		}

		/**
		 * Returns a string in lowercase.
		 *
		 * @param line string to process
		 * @return lowercase string
		 */
		SG_FORCED_INLINE std::string string_to_lower(const std::string& line)
		{
			std::string result;
			std::transform(
			    line.begin(), line.end(), std::back_inserter(result),
			    [](uint8_t val) { return std::tolower(val); });
			return result;
		}

		/**
		 * Returns string without whitespace
		 * @param line string to process
		 * @return string without whitespace
		 */
		SG_FORCED_INLINE std::string remove_whitespace(const std::string& line)
		{
			std::string result = line;
			result.erase(
			    std::remove_if(result.begin(), result.end(), ::isspace),
			    result.end());
			return result;
		}

		/**
		 * Removes all occurences of a character in place
		 * @param line string to process
		 * @param character char to remove
		 */
		SG_FORCED_INLINE void
		remove_char_inplace(std::string& line, char character)
		{
			line.erase(
			    std::remove_if(
			        line.begin(), line.end(),
			        [&character](auto const& val) { return val == character; }),
			    line.end());
		}
	} // namespace arff_detail
	/**
	 * ARFFDeserializer parses files in the ARFF format.
	 * For information about this format see
	 * https://waikato.github.io/weka-wiki/arff_stable/
	 */
	class ARFFDeserializer
	{
	public:
		/**
		 * ARFFDeserializer constructor with a filename.
		 * Performs a check to see if a file can be streamed.
		 * Fails if file does not exist, or it cannot be opened,
		 * i.e. not the correct permission.
		 *
		 * @param filename the name of the file to parse
		 */
		explicit ARFFDeserializer(const std::string& filename)
		{
			m_file_stream = std::ifstream(filename);
			if (m_file_stream.fail())
			{
				SG_SERROR("Cannot open %s\n", filename.c_str())
			}
		}

		/**
		 * Parse the file passed to the contructor.
		 *
		 */
		void read();

		/**
		 * Returns the data processed after parsing.
		 * @return matrix with parsed data
		 */
		SGMatrix<float64_t> get_data()
		{
			return m_data_matrix;
		}

	private:
		/**
		 * Processes a chunk. A chunk is defined as a set of lines that
		 * are processed in the same way. A chunk ends when the func
		 * sets the internal m_state to false.
		 * Parsing can also end when the stream reaches EOF.
		 *
		 * @tparam LambdaT type of processing function
		 * @tparam CheckT type of check function
		 * @param func processing function that reads each line
		 * @param check_func function that checks the result from the processing
		 * function
		 * @param skip_first whether to stream the first line
		 */
		template <typename LambdaT, typename CheckT>
		void process_chunk(LambdaT&& func, CheckT&& check_func, bool skip_first)
		{
			m_state = false;

			if (skip_first && !m_file_stream.eof())
				func();

			while (!m_state && !m_file_done)
			{
				consume_line(func);
			}
			if (!check_func())
			{
				SG_SERROR("Parsing error: %d", m_current_line.c_str());
			}
		}

		/**
		 * Function called by process_chunk to process a "chunk" line by line.
		 * This function also checks if EOF has been reached.
		 *
		 * @tparam T type of processing function
		 * @param func line processing function
		 */
		template <typename T>
		void consume_line(T&& func)
		{
			if (m_file_stream.eof())
			{
				m_file_done = true;
				return;
			}
			std::getline(m_file_stream, m_current_line);
			m_line_number++;
			if (!arff_detail::string_is_blank(m_current_line))
				func();
		}

		/**
		 * Cleans up the tokens for nominal attributes.
		 *
		 * @param line the line with nominal attributes.
		 * @return returns a vector with the nominal values in the correct
		 * position.
		 */
		std::vector<std::string> clean_up(std::vector<std::string>& line);

		/** character used in file to comment out a line */
		static const char* m_comment_string;
		/** characters to declare relations, i.e. @relation */
		static const char* m_relation_string;
		/** characters to declare attributes, i.e. @attribute */
		static const char* m_attribute_string;
		/** characters to declare data fields, i.e. @data */
		static const char* m_data_string;

		/** internal line number counter for exceptions */
		size_t m_line_number;
		/** internal flag set true when string stream is EOF */
		bool m_file_done;
		/** internal state when set to true switches parsing rules */
		bool m_state;
		/** current row count of data */
		size_t m_row_count;
		/** the string after m_relation_string*/
		std::string m_relation;
		/** the shared file stream */
		std::ifstream m_file_stream;
		/** the string where comments are stored */
		std::vector<std::string> m_comments;
		/** the string representing the current line being parsed */
		std::string m_current_line;
		/** the attribute types in the order they are parsed */
		std::vector<std::string> m_attributes;
		/** the mapping of nominal attributes to their value */
		std::vector<std::pair<std::string, std::vector<std::string>>>
		    m_nominal_attributes;

		/** dynamic continuous vector with the parsed data */
		std::vector<float64_t> m_data;
		/** sgmatrix with the properly formatted data from m_data */
		SGMatrix<float64_t> m_data_matrix;
	};
} // namespace shogun

#endif // SHOGUN_ARFFFILE_H
