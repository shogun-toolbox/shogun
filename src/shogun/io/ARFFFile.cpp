/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/ARFFFile.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;
using namespace shogun::arff_detail;

const char* ARFFDeserializer::m_comment_string = "%";
const char* ARFFDeserializer::m_relation_string = "@RELATION";
const char* ARFFDeserializer::m_attribute_string = "@ATTRIBUTE";
const char* ARFFDeserializer::m_data_string = "@DATA";

std::vector<std::string>
ARFFDeserializer::clean_up(std::vector<std::string>& line)
{
	std::string result_string;
	std::vector<std::string> result;
	std::vector<std::string>::iterator begin;

	for (auto& elem : line)
	{
		elem.erase(
		    std::remove_if(
		        elem.begin(), elem.end(),
		        [](auto& v) { return v == ',' || v == '{' || v == '}'; }),
		    elem.end());
	}
	for (auto iter = line.begin(); iter != line.end(); ++iter)
	{
		if (iter->front() == '\'' || iter->front() == '\"')
		{
			result_string = *iter;
			if (iter->back() != '\'' && iter->back() != '\"')
			{
				begin = iter;
				++iter;
				while (iter->back() != '\'' && iter->back() != '\"')
				{
					if (iter == line.end())
					{
						SG_SERROR("Unbalanced quotes")
					}
					++iter;
				}
				// concatenate strings within quotes with a space in
				// between 
				result_string = std::accumulate(
				    begin + 1, iter + 1, *begin,
				    [](std::string s0, std::string& s1) {
					    remove_char_inplace(s0, '\'');
					    remove_char_inplace(s1, '\'');
					    return s0 += " " + s1;
				    });
			}
			else
				remove_char_inplace(result_string, '\'');
			result.push_back(result_string);
		}
		else
		{
			result_string = *iter;
			remove_char_inplace(result_string, '\'');
			if (!result_string.empty())
				result.push_back(result_string);
		}
	}
	return result;
}

void ARFFDeserializer::read()
{
	m_line_number = 0;
	m_row_count = 0;
	m_file_done = false;
	auto read_comment = [this]() {
		if (string_to_lower(m_current_line.substr(0, 1)) == m_comment_string)
			m_comments.push_back(m_current_line.substr(1, std::string::npos));
		else
			m_state = true;
	};
	auto check_comment = [this]() { return true; };
	process_chunk(read_comment, check_comment, false);

	auto read_relation = [this]() {
		if (string_to_lower(m_current_line.substr(
		        0, strlen(m_relation_string))) == m_relation_string)
		{
			m_relation = remove_whitespace(
			    m_current_line.substr(strlen(m_relation_string)));
		}
		else
			m_state = true;
	};
	// a relation has to be defined
	auto check_relation = [this]() { return !m_relation.empty(); };
	process_chunk(read_relation, check_relation, true);

	auto read_attributes = [this]() {
		if (string_to_lower(m_current_line.substr(
		        0, strlen(m_attribute_string))) == m_attribute_string)
		{
			std::vector<std::string> elems;
			auto innner_string =
			    m_current_line.substr(strlen(m_attribute_string));
			split(innner_string, " ,\t\r\f\v", std::back_inserter(elems));
			std::transform(
			    elems.begin(), elems.end(), elems.begin(),
			    [](const auto& val) { return remove_whitespace(val); });
			// check if it is nominal
			if (elems[1] == "{" || elems[1].front() == '{')
			{
				elems = clean_up(elems);
				std::vector<std::string> attributes(
				    elems.begin() + 1, elems.end());
				m_nominal_attributes.emplace_back(
				    std::make_pair(elems[0], attributes));
				m_attributes.emplace_back("nominal");
				return;
			}

			auto is_date = std::find(elems.begin(), elems.end(), "date");
			if (is_date != elems.end())
			{
				if (elems.begin() == is_date && elems.size() < 2)
				{
					// TODO: @attribute date [[date-format]]
				}
				else if (elems.begin() + 1 == is_date && elems.size() < 3)
				{
					// TODO: @attribute [name] date [[date-format]]
				}
				else
				{
					SG_SERROR("Error parsing date on line %d", m_line_number)
				}
				// m_attributes.emplace(std::make_pair(elems[0],
				// "date"));
				m_attributes.emplace_back("date");
			}
			else if (elems.size() == 2)
			{
				auto type = string_to_lower(elems[1]);
				// numeric attributes
				if (type == "numeric" || type == "integer" || type == "real")
				{
					// m_attributes.emplace(std::make_pair(elems[0],
					// "numeric"));
					m_attributes.emplace_back("numeric");
				}
				else if (type == "string")
				{
					// @ATTRIBUTE LCC    string
					// m_attributes.emplace(std::make_pair(elems[0],
					// "string"));
					m_attributes.emplace_back("string");
				}
				else
					SG_SERROR(
					    "Unexpected attribute type identifier \"%s\" "
					    "on line %d\n",
					    type.c_str(), m_line_number)
			}
			else
				SG_SERROR(
				    "Unexpected format in @ATTRIBUTE on line %d\n",
				    m_line_number);
		}
		// comments in this section are ignored
		else if (m_current_line.substr(0, 1) == m_comment_string)
		{
			return;
		}
		// if none of the others are true this is the end of the
		// attributes section
		else
		{
			m_state = true;
		}
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
		// it's the data string (i.e. @data"), does not provide
		// information
		if (string_to_lower(m_current_line.substr(0, strlen(m_data_string))) ==
		    m_data_string)
		{
			return;
		}
		else
		{
			std::vector<std::string> elems;
			std::string type;
			split(m_current_line, ",", std::back_inserter(elems));
			auto nominal_pos = m_nominal_attributes.begin();
			for (int i = 0; i < elems.size(); ++i)
			{
				type = m_attributes[i];
				if (type == "numeric")
				{
					m_data.push_back(std::stod(elems[i]));
				}
				else if (type == "nominal")
				{
					if (nominal_pos == m_nominal_attributes.end())
						SG_SERROR(
						    "Unexpected nominal value \"%s\" on line "
						    "%d\n",
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
