/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/io/ARFFFile.h>

#include <date/date.h>

using namespace shogun;
using namespace shogun::arff_detail;

const char* ARFFDeserializer::m_comment_string = "%";
const char* ARFFDeserializer::m_relation_string = "@relation";
const char* ARFFDeserializer::m_attribute_string = "@attribute";
const char* ARFFDeserializer::m_data_string = "@data";
const char* ARFFDeserializer::m_default_date_format = "%Y-%M-%DT%H:%M:%S";

void ARFFDeserializer::read()
{
	m_line_number = 0;
	m_row_count = 0;
	m_file_done = false;
	auto read_comment = [this]() {
		if (to_lower(m_current_line.substr(0, 1)) == m_comment_string)
			m_comments.push_back(m_current_line.substr(1, std::string::npos));
		else if (
		    to_lower(m_current_line.substr(0, strlen(m_relation_string))) ==
		    m_relation_string)
			m_state = true;
	};
	auto check_comment = []() { return true; };
	process_chunk(read_comment, check_comment, false);

	auto read_relation = [this]() {
		if (to_lower(m_current_line.substr(0, strlen(m_relation_string))) ==
		    m_relation_string)
			m_relation = remove_whitespace(
			    m_current_line.substr(strlen(m_relation_string)));
		else if (
		    to_lower(m_current_line.substr(0, strlen(m_attribute_string))) ==
		    m_attribute_string)
			m_state = true;
	};
	// a relation has to be defined
	auto check_relation = [this]() { return !m_relation.empty(); };
	process_chunk(read_relation, check_relation, true);

	auto read_attributes = [this]() {
		if (to_lower(m_current_line.substr(0, strlen(m_attribute_string))) ==
		    m_attribute_string)
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
				// @ATTRIBUTE class {Iris-setosa,Iris-versicolor,Iris-virginica}
				std::vector<std::string> attributes;
				// split norminal values: "{A, B, C}" to vector{A, B, C}
				split(
				    type.substr(1, type.size() - 2), ", ",
				    std::back_inserter(attributes), "\'\"");
				m_nominal_attributes.emplace_back(
				    std::make_pair(name, attributes));
				m_attributes.push_back(Attribute::Nominal);
				m_data_vectors.emplace_back(std::vector<float64_t>{});
				m_attribute_names.emplace_back(name);
				return;
			}

			auto is_date = type.find("date") != std::string::npos;
			if (is_date)
			{
				std::vector<std::string> date_elements;
				// split "date [[date-format]]" or "name date [[date-format]]"
				split(type, " ", std::back_inserter(date_elements), "\"");
				if (date_elements[0] == "date" && date_elements.size() < 3)
				{
					// @attribute date [[date-format]]
					if (type.size() == 1)
						m_date_formats.emplace_back(m_default_date_format);
					else
						m_date_formats.push_back(
						    javatime_to_cpptime(date_elements[1]));
					name = "";
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
				m_data_vectors.emplace_back(std::vector<float64_t>{});
			}
			else if (is_primitive_type(type))
			{
				type = to_lower(type);
				// numeric attributes
				if (type == "numeric")
				{
					m_attributes.push_back(Attribute::Numeric);
					m_data_vectors.emplace_back(std::vector<float64_t>{});
				}
				else if (type == "integer")
				{
					m_attributes.push_back(Attribute::Integer);
					m_data_vectors.emplace_back(std::vector<float64_t>{});
				}
				else if (type == "real")
				{
					m_attributes.push_back(Attribute::Real);
					m_data_vectors.emplace_back(std::vector<float64_t>{});
				}
				else if (type == "string")
				{
					// @ATTRIBUTE LCC    string
					m_attributes.push_back(Attribute::String);
					m_data_vectors.emplace_back(std::vector<std::string>{});
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
			m_attribute_names.emplace_back(name);
		}
		// comments in this section are ignored
		else if (m_current_line.substr(0, 1) == m_comment_string)
		{
		}
		else if (
		    to_lower(m_current_line.substr(0, strlen(m_data_string))) ==
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
		if (SG_UNLIKELY(m_current_line.substr(0, 1) == m_comment_string))
			return;
		// it's the data string (i.e. @data"), does not provide information
		if (SG_UNLIKELY(
		        to_lower(m_current_line.substr(0, strlen(m_data_string))) ==
		        m_data_string))
		{
			return;
		}
		// assumes that until EOF we should expect comma delimited values
		std::vector<std::string> elems;
		split(m_current_line, ",", std::back_inserter(elems), "\'\"");
		// only parse rows that do not contain missing values
		if (std::find(elems.begin(), elems.end(), "?") == elems.end())
		{
			auto nominal_pos = m_nominal_attributes.begin();
			auto date_pos = m_date_formats.begin();
			int i = 0;
			for (; i < elems.size(); ++i)
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
						shogun::get<std::vector<float64_t>>(m_data_vectors[i])
						    .push_back(std::stod(elems[i]));
					}
					catch (const std::invalid_argument&)
					{
						SG_SERROR(
						    "Failed to covert \"%s\" to numeric on line %d.\n",
						    elems[i].c_str(), m_line_number)
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
					auto trimmed_el = trim(elems[i]);
					remove_char_inplace(trimmed_el, '\'');
					auto pos =
					    std::find(encoding.begin(), encoding.end(), trimmed_el);
					if (pos == encoding.end())
						SG_SERROR(
						    "Unexpected value \"%s\" on line %d\n",
						    trimmed_el.c_str(), m_line_number);
					float64_t idx = std::distance(encoding.begin(), pos);
					shogun::get<std::vector<float64_t>>(m_data_vectors[i])
					    .push_back(idx);
					nominal_pos = std::next(nominal_pos);
				}
				break;
				case (Attribute::Date):
				{
					date::sys_seconds t;
					std::istringstream ss(elems[i]);
					if (date_pos == m_date_formats.end())
						SG_SERROR(
						    "Unexpected date value \"%s\" on line %d.\n",
						    elems[i].c_str(), m_line_number);
					ss >> date::parse(*date_pos, t);
					if (bool(ss))
					{
						auto value_timestamp = t.time_since_epoch().count();
						shogun::get<std::vector<float64_t>>(m_data_vectors[i])
						    .push_back(value_timestamp);
					}
					else
						SG_SERROR(
						    "Error parsing date \"%s\" with date format \"%s\" "
						    "on line %d.\n",
						    elems[i].c_str(), (*date_pos).c_str(),
						    m_line_number)
					++date_pos;
				}
				break;
				case (Attribute::String):
					shogun::get<std::vector<std::string>>(m_data_vectors[i])
					    .emplace_back(elems[i]);
				}
			}
			if (i != m_attributes.size())
				SG_SERROR(
				    "Unexpected number of values on line %d, expected %d "
				    "values, "
				    "but found %d.\n",
				    m_line_number, m_attributes.size(), i)
			++m_row_count;
		}
	};
	auto check_data = [this]() {
		// check X values
		SG_SDEBUG(
		    "size: %d, cols: %d, rows: %d", m_data_vectors.size(),
		    m_data_vectors.size() / m_row_count, m_row_count)
		if (!m_data_vectors.empty())
		{
			auto feature_count = m_data_vectors.size();
			index_t row_count =
			    shogun::visit(VectorSizeVisitor{}, m_data_vectors[0]);
			for (int i = 1; i < feature_count; ++i)
			{
				REQUIRE(
				    shogun::visit(VectorSizeVisitor{}, m_data_vectors[i]) ==
				        row_count,
				    "All columns must have the same number of features!\n")
			}
		}
		else
			return false;
		return true;
	};
	process_chunk(read_data, check_data, true);
}

std::shared_ptr<CCombinedFeatures> ARFFDeserializer::get_features() const
{
	auto result = std::make_shared<CCombinedFeatures>();
	index_t row_count = shogun::visit(VectorSizeVisitor{}, m_data_vectors[0]);
	for (int i = 0; i < m_data_vectors.size(); ++i)
	{
		Attribute att = m_attributes[i];
		auto vec = m_data_vectors[i];
		switch (att)
		{
		case Attribute::Numeric:
		case Attribute::Integer:
		case Attribute::Real:
		case Attribute::Date:
		case Attribute::Nominal:
		{
			auto casted_vec = shogun::get<std::vector<float64_t>>(vec);
			SGMatrix<float64_t> mat(1, row_count);
			memcpy(
			    mat.matrix, casted_vec.data(),
			    casted_vec.size() * sizeof(float64_t));
			auto* feat = new CDenseFeatures<float64_t>(mat);
			result->append_feature_obj(feat);
		}
		break;
		case Attribute::String:
		{
			auto casted_vec = shogun::get<std::vector<std::string>>(vec);
			index_t max_string_length = 0;
			for (const auto& el : casted_vec)
			{
				if (max_string_length < el.size())
					max_string_length = el.size();
			}
			SGStringList<char> strings(row_count, max_string_length);
			for (int j = 0; j < row_count; ++j)
			{
				SGString<char> current(max_string_length);
				memcpy(
				    current.string, casted_vec[j].data(),
				    (casted_vec.size() + 1) * sizeof(char));
				strings.strings[j] = current;
			}
			auto* feat = new CStringFeatures<char>(strings, EAlphabet::RAWBYTE);
			result->append_feature_obj(feat);
		}
		}
	}
	return result;
}
