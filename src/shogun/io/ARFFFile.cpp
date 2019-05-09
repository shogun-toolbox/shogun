/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/ARFFFile.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <ctime>

using namespace shogun;
using namespace shogun::arff_detail;

const char* ARFFDeserializer::m_comment_string = "%";
const char* ARFFDeserializer::m_relation_string = "@relation";
const char* ARFFDeserializer::m_attribute_string = "@attribute";
const char* ARFFDeserializer::m_data_string = "@data";
const char* ARFFDeserializer::m_default_date_format = "%Y-%M-%D %Z %H:%M:%S";

void ARFFDeserializer::read()
{
	m_line_number = 0;
	m_row_count = 0;
	m_file_done = false;
	auto read_comment = [this]() {
		if (string_to_lower(m_current_line.substr(0, 1)) == m_comment_string)
			m_comments.push_back(m_current_line.substr(1, std::string::npos));
		else if (
		    string_to_lower(m_current_line.substr(
		        0, strlen(m_relation_string))) == m_relation_string)
			m_state = true;
	};
	auto check_comment = []() { return true; };
	process_chunk(read_comment, check_comment, false);

	auto read_relation = [this]() {
		if (string_to_lower(m_current_line.substr(
		        0, strlen(m_relation_string))) == m_relation_string)
			m_relation = remove_whitespace(
			    m_current_line.substr(strlen(m_relation_string)));
		else if (
		    string_to_lower(m_current_line.substr(
		        0, strlen(m_attribute_string))) == m_attribute_string)
			m_state = true;
	};
	// a relation has to be defined
	auto check_relation = [this]() { return !m_relation.empty(); };
	process_chunk(read_relation, check_relation, true);

	auto read_attributes = [this]() {
		if (string_to_lower(m_current_line.substr(
		        0, strlen(m_attribute_string))) == m_attribute_string)
		{
			// store attribute name and type
			std::string name;
			std::string type;
			auto inner_string =
			    m_current_line.substr(strlen(m_attribute_string));
			left_trim(inner_string);
			auto it = inner_string.begin();
			while (it != inner_string.end())
			{
				if (!std::isspace(*it))
					it = std::next(it);
				else
				{
					name = trim({inner_string.begin(), it});
					type = trim({it, inner_string.end()});
					break;
				}
			}
			if (it == inner_string.end())
				SG_SERROR(
				    "Could not split attibute name and type on line %d: "
				    "\"%s\".\n",
				    m_line_number, m_current_line.c_str())
			// check if it is nominal
			if (type[0] == '{')
			{
				std::vector<std::string> attributes;
				// split norminal values: "{A, B, C}" to vector{A, B, C}
				split(
				    type.substr(1, type.size() - 2), ", ", true,
				    std::back_inserter(attributes));
				m_nominal_attributes.emplace_back(
				    std::make_pair(name, attributes));
				m_attributes.push_back(Attribute::Nominal);
				return;
			}

			auto is_date = type.find("date") != std::string::npos;
			if (is_date)
			{
				std::vector<std::string> date_elements;
				// split "date [[date-format]]" or "name date [[date-format]]"
				split(type, " ", true, std::back_inserter(date_elements));
				if (date_elements[0] == "date" && date_elements.size() < 3)
				{
					// @attribute date [[date-format]]
					if (type.size() == 1)
						m_date_formats.emplace_back(m_default_date_format);
					else
						m_date_formats.push_back(
						    javatime_to_cpptime(date_elements[1]));
				}
				else if (date_elements[1] == "date" && date_elements.size() < 4)
				{
					// @attribute name date [[date-format]]
					if (date_elements.size() == 2)
						m_date_formats.emplace_back(m_default_date_format);
					else
						m_date_formats.push_back(
						    javatime_to_cpptime(date_elements[2]));
				}
				else
				{
					SG_SERROR(
					    "Error parsing date on line %d: %s\n", m_line_number,
					    m_current_line.c_str())
				}
				m_attributes.push_back(Attribute::Date);
			}
			else if (is_primitive_type(type))
			{
				type = string_to_lower(type);
				// numeric attributes
				if (type == "numeric")
					m_attributes.push_back(Attribute::Numeric);
				else if (type == "integer")
					m_attributes.push_back(Attribute::Integer);
				else if (type == "real")
					m_attributes.push_back(Attribute::Real);
				else if (type == "string")
				{
					// @ATTRIBUTE LCC    string
					// m_attributes.emplace(std::make_pair(elems[0],
					// "string"));
					m_attributes.push_back(Attribute::String);
				}
				else
					SG_SERROR(
					    "Unexpected attribute type identifier \"%s\" "
					    "on line %d: %s\n",
					    type.c_str(), m_line_number, m_current_line.c_str())
			}
			else
				SG_SERROR(
				    "Unexpected format in @ATTRIBUTE on line %d: %s\n",
				    m_line_number, m_current_line.c_str());
		}
		// comments in this section are ignored
		else if (m_current_line.substr(0, 1) == m_comment_string)
		{
		}
		else if (
		    string_to_lower(m_current_line.substr(0, strlen(m_data_string))) ==
		    m_data_string)
			m_state = true;
	};

	auto check_attributes = [this]() {
		// attributes cannot be empty
		return !m_attributes.empty();
	};
	process_chunk(read_attributes, check_attributes, true);

	auto read_data = [this]() {
		// it's a comment and can be skipped
		if (m_current_line.substr(0, 1) == m_comment_string)
			return;
		// it's the data string (i.e. @data"), does not provide information
		if (string_to_lower(m_current_line.substr(0, strlen(m_data_string))) ==
		    m_data_string)
		{
			return;
		}
		// assumes that until EOF we should expect comma delimited values
		std::vector<std::string> elems;
		split(m_current_line, ",", true, std::back_inserter(elems));
		auto nominal_pos = m_nominal_attributes.begin();
		auto date_pos = m_date_formats.begin();
		for (int i = 0; i < elems.size(); ++i)
		{
			Attribute type = m_attributes[i];
			switch (type)
			{
			case (Attribute::Numeric):
			case (Attribute::Integer):
			case (Attribute::Real):
			{
				try
				{
					m_data.push_back(std::stod(elems[i]));
				}
				catch (const std::invalid_argument&)
				{
					SG_SERROR(
					    "Failed to covert \"%s\" to numeric.\n",
					    elems[i].c_str())
				}
			}
			break;
			case (Attribute::Nominal):
			{
				if (nominal_pos == m_nominal_attributes.end())
					SG_SERROR(
					    "Unexpected nominal value \"%s\" on line %d\n",
					    elems[i].c_str(), m_line_number);
				auto encoding = (*nominal_pos).second;
				remove_char_inplace(elems[i], '\'');
				auto pos =
				    std::find(encoding.begin(), encoding.end(), elems[i]);
				if (pos == encoding.end())
					SG_SERROR(
					    "Unexpected value \"%s\" on line %d\n",
					    elems[i].c_str(), m_line_number);
				float64_t idx = std::distance(encoding.begin(), pos);
				m_data.push_back(idx);
				nominal_pos = std::next(nominal_pos);
			}
			break;
			case (Attribute::Date):
			{
				tm t{};
				if (date_pos == m_date_formats.end())
					SG_SERROR(
					    "Unexpected date value \"%s\" on line %d.\n",
					    elems[i].c_str(), m_line_number);
				if (strptime(elems[i].c_str(), (*date_pos).c_str(), &t))
				{
					auto value_timestamp = std::mktime(&t);
					if (value_timestamp == -1)
						SG_SERROR(
						    "Error creating timestamp with \"%s\" with "
						    "date format \"%s\" on line %d.\n",
						    elems[i].c_str(), (*date_pos).c_str(),
						    m_line_number)
					else
						m_data.emplace_back(value_timestamp);
				}
				else
					SG_SERROR(
					    "Error parsing date \"%s\" with date format \"%s\" "
					    "on line %d.\n",
					    elems[i].c_str(), (*date_pos).c_str(), m_line_number)
				++date_pos;
			}
			break;
			case (Attribute::String):
				SG_SERROR("String parsing not implemented.\n")
			}
		}
		++m_row_count;
	};
	auto check_data = [this]() {
		// check X values
		SG_SDEBUG(
		    "size: %d, cols: %d, rows: %d", m_data.size(),
		    m_data.size() / m_row_count, m_row_count)
		if (!m_data.empty())
		{
			auto tmp =
			    SGMatrix<float64_t>(m_data.size() / m_row_count, m_row_count);
			m_data_matrix =
			    SGMatrix<float64_t>(m_row_count, m_data.size() / m_row_count);
			memcpy(
			    tmp.matrix, m_data.data(), m_data.size() * sizeof(float64_t));
			typename SGMatrix<float64_t>::EigenMatrixXtMap tmp_eigen = tmp;
			typename SGMatrix<float64_t>::EigenMatrixXtMap m_data_matrix_eigen =
			    m_data_matrix;

			m_data_matrix_eigen = tmp_eigen.transpose();
		}
		else
			return false;
		return true;
	};
	process_chunk(read_data, check_data, true);
}
