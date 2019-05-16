/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ARFFFILE_H
#define SHOGUN_ARFFFILE_H

#include <shogun/base/init.h>
#include <shogun/base/variant.h>
#include <shogun/features/CombinedFeatures.h>
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
#ifndef SWIG
	/** Contains miscellaneous string manipulation functions using the STL and
	 * Java to C++ date format utilities */
	namespace arff_detail
	{
		/**
		 * Checks if string is blank
		 * @param line to check
		 * @return bool whether line is empty
		 */
		SG_FORCED_INLINE bool is_blank(const std::string& line)
		{
			return line.find_first_not_of(" \t\r\f\v") == std::string::npos;
		}

		/**
		 * Checks if the character in the lhs is in the rhs
		 * @param lhs the single character to find
		 * @param rhs a string with the characters to compare against the lhs
		 * @return whether the character of the lhs is in the rhs
		 */
		SG_FORCED_INLINE bool contains(char lhs, const std::string& rhs)
		{
			auto result = rhs.find(lhs);
			return result != std::string::npos;
		}

		/**
		 * Trims the whitespace to the left of a string in place.
		 * @param s the string to trim
		 */
		SG_FORCED_INLINE void left_trim(std::string& s)
		{
			s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](char val) {
				        return !std::isspace(val);
			        }));
		}

		/**
		 * Trims the whitespace to the right of a string in place.
		 * @param s the string to trim
		 */
		SG_FORCED_INLINE void right_trim(std::string& s)
		{
			s.erase(
			    std::find_if(
			        s.rbegin(), s.rend(),
			        [](char val) { return !std::isspace(val); })
			        .base(),
			    s.end());
		}

		/**
		 * Returns the string with any whitespace to the left and right trimmed.
		 * @param s the string to trim
		 */
		SG_FORCED_INLINE std::string trim(std::string line)
		{
			left_trim(line);
			right_trim(line);
			return line;
		}

		/**
		 * Splits a line given a set of delimiter characters
		 *
		 * @param s string to split
		 * @param delimiters a set of delimiter character
		 * @param result dynamic container where tokens are stored
		 * @param quotes a string with the characters that are considered
		 * quotes, i.e. any text between quotes is kept together.
		 */
		SG_FORCED_INLINE void split(
		    const std::string& s, const std::string& delimiters,
		    std::back_insert_iterator<std::vector<std::string>> result,
		    const std::string& quotes)
		{
			auto it = s.begin();
			auto begin = s.begin();
			while (arff_detail::contains(*it, delimiters))
			{
				it = std::next(it);
				begin = it;
			}
			while (it != s.end())
			{
				if (arff_detail::contains(*it, delimiters))
				{
				}
				else if (arff_detail::contains(*it, quotes))
				{
					begin = std::next(it);
					it = begin;
					while (!arff_detail::contains(*it, quotes))
					{
						it = std::next(it);
					}
					if (it == s.end())
						SG_SERROR(
						    "Encountered unbalanced parenthesis in \"%s\"\n",
						    std::string(std::prev(begin), it).c_str())
					*(result++) = {begin, it};
				}
				else
				{
					begin = it;
					while (!arff_detail::contains(*it, delimiters) &&
					       it != s.end())
					{
						it = std::next(it);
					}
					auto token = std::string(begin, it);
					if (!arff_detail::is_blank(token))
						*(result++) = token;
				}
				if (it != s.end())
				{
					begin = std::next(it);
					it = std::next(it);
				}
			}
		}

		/**
		 * Returns a string in lowercase.
		 *
		 * @param line string to process
		 * @return lowercase string
		 */
		SG_FORCED_INLINE std::string to_lower(const std::string& line)
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

		/**
		 * Java to C++ time format token converter.
		 *
		 * Java tokens taken from:
		 * http://tutorials.jenkov.com/java-date-time/parsing-formatting-dates.html
		 * C++ tokens taken from:
		 * https://www.ibm.com/support/knowledgecenter/en/ssw_ibm_i_71/rtref/strpti.htm
		 * @param java_token
		 * @return C++ equivalent
		 */
		SG_FORCED_INLINE const char*
		process_javatoken(const std::string& java_token)
		{
			if (java_token == "yy")
				return "%y";
			if (java_token == "yyyy")
				return "%Y";
			if (java_token == "MM")
				return "%m";
			if (java_token == "dd")
				return "%d";
			if (java_token == "hh")
				return "%I";
			if (java_token == "HH")
				return "%H";
			if (java_token == "mm")
				return "%M";
			if (java_token == "ss")
				return "%S";
			if (java_token == "Z")
				return "%z";
			if (java_token == "z")
				SG_SERROR(
				    "Timezone abbreviations are currently not supported.\n")
			if (java_token.empty())
				return "";
			if (java_token == "SSS")
				return nullptr;
			return nullptr;
		}
		SG_FORCED_INLINE const char* process_javatoken(char java_token)
		{
			if (java_token == ':')
				return ":";
			if (java_token == '\'')
				return "";
			if (java_token == '-')
				return "-";
			if (java_token == ' ')
				return " ";
			return nullptr;
		}

		/**
		 * Checks if a Java token is valid and returns string representing the
		 * C++ token.
		 * @param java_time_token token to check and translate
		 * @return translated C++ token
		 */
		SG_FORCED_INLINE const char*
		check_and_append_j2cpp(const std::string& java_time_token)
		{
			if (auto cpp_token = process_javatoken(java_time_token))
				return cpp_token;
			else
				SG_SERROR(
				    "Could not convert Java time token \"%s\" to C++ time "
				    "token.\n",
				    java_time_token.c_str())
			return nullptr;
		}

		SG_FORCED_INLINE const char*
		check_and_append_j2cpp(char java_time_token)
		{
			if (auto cpp_token = process_javatoken(java_time_token))
				return cpp_token;
			else
				SG_SERROR(
				    "Could not convert Java time token \"%c\" to C++ time "
				    "token.\n",
				    java_time_token)
			return nullptr;
		}

		/**
		 * Converts a Java SimpleDateFormat to a C++ date format
		 * @param java_time the string to translate
		 * @return the C++ format equivalent of java_time
		 */
		SG_FORCED_INLINE std::string
		javatime_to_cpptime(const std::string& java_time)
		{
			std::string cpp_time;
			std::string token;
			auto begin = java_time.begin();
			auto it = java_time.begin();
			while (it != java_time.end())
			{
				if (*it == '-' || *it == ' ' || *it == ':')
				{
					token = {begin, it};
					cpp_time.append(check_and_append_j2cpp(token));
					cpp_time.append(check_and_append_j2cpp(*it));
					begin = std::next(it);
				}
				else if (*it == '\'')
				{
					token = {begin, it};
					cpp_time.append(check_and_append_j2cpp(token));
					cpp_time.append(check_and_append_j2cpp(*it));
					begin = std::next(it);
					begin = it;
					it = std::next(it);
					while (*it != '\'')
					{
						it = std::next(it);
					}
					token = {std::next(begin), it};
					cpp_time.append(token);
					begin = std::next(it);
				}
				else if (std::next(it) == java_time.end())
				{
					token = {begin, std::next(it)};
					if (auto cpp_token = process_javatoken(token))
					{
						cpp_time.append(cpp_token);
					}
					else
						SG_SERROR(
						    "Could not convert Java time token %s to C++ time "
						    "token.\n",
						    token.c_str())
				}
				it = std::next(it);
			}
			return cpp_time;
		}
	}  // namespace arff_detail
#endif // SWIG
	/**
	 * ARFFDeserializer parses files in the ARFF format.
	 * For information about this format see
	 * https://waikato.github.io/weka-wiki/arff_stable/
	 */
	class ARFFDeserializer
	{
	private:
		/**
		 * The attributes supported in the ARFF format
		 */
		enum class Attribute
		{
			Numeric = 0,
			Integer = 1,
			Real = 2,
			String = 3,
			Date = 4,
			Nominal = 5
		};

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
			auto* file_stream = new std::ifstream(filename);
			if (file_stream->fail())
			{
				SG_SERROR(
				    "Cannot open %s. Please check if file exists and if you "
				    "have the right permissions to open it.\n",
				    filename.c_str())
			}
			m_stream = std::unique_ptr<std::istream>(file_stream);
		}
#ifndef SWIG
		/**
		 * ARFFDeserializer constructor with an input stream.
		 * This constructors copies the stream and takes care
		 * of proper deletion.
		 *
		 * @param filename the input stream
		 */
		explicit ARFFDeserializer(std::shared_ptr<std::istream>& stream)
		{
			m_stream = stream;
		}
#endif // SWIG
		/**
		 * Parse the file passed to the contructor.
		 *
		 */
		void read();

		/**
		 * Returns string parsed in @relation line
		 * @return the relation string
		 */
		std::string get_relation() const noexcept
		{
			return m_relation;
		}

		/**
		 * Returns the name of the features parsed in @attribute
		 * @return the relation string
		 */
		std::vector<std::string> get_feature_names() const noexcept
		{
			return m_attribute_names;
		}

		/**
		 * Get combined features from parsed data
		 * @return
		 */
		std::shared_ptr<CCombinedFeatures> get_features() const;

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

			if (skip_first && !m_stream->eof())
				func();

			while (!m_state && !m_file_done)
			{
				consume_line(func);
			}
			if (!check_func())
			{
				SG_SERROR(
				    "Parsing error on line %d: %s\n", m_line_number,
				    m_current_line.c_str());
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
			if (m_stream->eof())
			{
				m_file_done = true;
				return;
			}
			std::getline(*m_stream, m_current_line);
			m_line_number++;
			if (!arff_detail::is_blank(m_current_line))
				func();
		}
#ifndef SWIG
		/**
		 * Checks if a token represented by a string
		 * denotes a primitive type in the ARFF format
		 * @param token the token to be checked
		 * @return whether the token denotes a primitive type
		 */
		SG_FORCED_INLINE bool is_primitive_type(const std::string& token) const
		    noexcept
		{
			return token.find_first_of("numeric") != std::string::npos ||
			       token.find_first_of("integer") != std::string::npos ||
			       token.find_first_of("real") != std::string::npos ||
			       token.find_first_of("string") != std::string::npos;
		}
#endif // SWIG

		void reserve_vector_memory(size_t line_count)
		{
			VectorResizeVisitor visitor{line_count};
			for (auto& vec : m_data_vectors)
				shogun::visit(visitor, vec);
		}

		/**
		 * Visitor pattern to reserve memory for a std::vector
		 * wrapped in a variant class.
		 */
		struct VectorResizeVisitor
		{
			VectorResizeVisitor(size_t size) : m_size(size){};
			template <typename T>
			void operator()(std::vector<T>& v) const noexcept
			{
				v.reserve(m_size);
			}
			int m_size;
		};

		/**
		 * Visitor pattern to determine size of a std::vector
		 * wrapped in a variant class.
		 */
		struct VectorSizeVisitor
		{
			template <typename T>
			size_t operator()(const std::vector<T>& v) const noexcept
			{
				return v.size();
			}
		};

		/** character used in file to comment out a line */
		static const char* m_comment_string;
		/** characters to declare relations, i.e. @relation */
		static const char* m_relation_string;
		/** characters to declare attributes, i.e. @attribute */
		static const char* m_attribute_string;
		/** characters to declare data fields, i.e. @data */
		static const char* m_data_string;
		/** the default C++ date format specified by the ARFF standard */
		static const char* m_default_date_format;
		/** the name of the attributes */
		std::vector<std::string> m_attribute_names;

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
		/** the input stream */
		std::shared_ptr<std::istream> m_stream;
		/** the string where comments are stored */
		std::vector<std::string> m_comments;
		/** the string representing the current line being parsed */
		std::string m_current_line;
		/** the attribute types in the order they are parsed */
		std::vector<Attribute> m_attributes;
		/** stores the date formats */
		std::vector<std::string> m_date_formats;
		/** the mapping of nominal attributes to their value */
		std::vector<std::pair<std::string, std::vector<std::string>>>
		    m_nominal_attributes;

		/** a vector containing the feature vectors containing the parsed data
		 * in the correct type
		 */
		std::vector<variant<std::vector<float64_t>, std::vector<std::string>>>
		    m_data_vectors;
	};
} // namespace shogun

#endif // SHOGUN_ARFFFILE_H
