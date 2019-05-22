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
const char* ARFFDeserializer::m_missing_value_string = "?";

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
	size_t m_size;
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

template <typename T>
T buffer_to_type(const std::string& buffer)
{
	SG_SERROR(
	    "No conversion from \"%s\" to \"%s\"!\n", buffer.c_str(),
	    demangled_type<T>())
}
template <>
int8_t buffer_to_type<int8_t>(const std::string& buffer)
{
	return static_cast<int8_t>(std::stoi(buffer));
}
template <>
int16_t buffer_to_type<int16_t>(const std::string& buffer)
{
	return static_cast<int16_t>(std::stoi(buffer));
}
template <>
int32_t buffer_to_type<int32_t>(const std::string& buffer)
{
	return std::stoi(buffer);
}
template <>
int64_t buffer_to_type<int64_t>(const std::string& buffer)
{
	return std::stoll(buffer);
}
template <>
float32_t buffer_to_type<float32_t>(const std::string& buffer)
{
	return std::stof(buffer);
}
template <>
float64_t buffer_to_type<float64_t>(const std::string& buffer)
{
	return std::stod(buffer);
}
template <>
floatmax_t buffer_to_type<floatmax_t>(const std::string& buffer)
{
	return std::stold(buffer);
}

template <typename ScalarType, typename CharType>
void ARFFDeserializer::read_helper()
{
	std::vector<variant<
	    std::vector<ScalarType>, std::vector<std::basic_string<CharType>>>>
	    data_vectors;
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

	// parse the @attributes section
	auto read_attributes = [this, &data_vectors]() {
		if (to_lower(m_current_line.substr(0, strlen(m_attribute_string))) ==
		    m_attribute_string)
		{
			std::string name, type;
			auto inner_string =
			    m_current_line.substr(strlen(m_attribute_string));
			left_trim(
			    inner_string, [](const auto& val) { return !std::isspace(val); });
			auto it = inner_string.begin();
			if (is_part_of(*it, "\"\'"))
			{
				auto quote_type = *it;
				++it;
				auto begin = it;
				while (*it != quote_type && it != inner_string.end())
					++it;
				if (it == inner_string.end())
					SG_SERROR(
					    "Encountered unbalanced parenthesis in attribute "
					    "declaration on line %d: \"%s\"\n",
					    m_line_number, m_current_line.c_str())
				name = {begin, it};
				type = trim({std::next(it), inner_string.end()});
			}
			else
			{
				auto begin = it;
				while (!std::isspace(*it))
					++it;
				if (it == inner_string.end() && it != inner_string.end())
					SG_SERROR(
					    "Expected at least two elements in attribute "
					    "declaration on line %d: \"%s\"",
					    m_line_number, m_current_line.c_str())
				name = {begin, it};
				type = trim({std::next(it), inner_string.end()});
			}

			SG_SDEBUG("name: %s\n", name.c_str())
			SG_SDEBUG("type: %s\n", type.c_str())

			if (name.empty() || type.empty())
				SG_SERROR(
				    "Could not find the name and type on line %d: \"%s\".\n",
				    m_line_number, m_current_line.c_str())
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
				// split nominal values: "{A, B, C}" to vector{A, B, C}
				split(
				    type.substr(1, type.size() - 2), ", ",
				    std::back_inserter(attributes), "\'\"");
				auto processed_name = trim(name, [](const auto& val) {
					return !std::isspace(val) && val != '\'' && val != '\"';
				});
				m_attribute_names.emplace_back(processed_name);
				m_nominal_attributes.emplace_back(
				    std::make_pair(name, attributes));
				m_attributes.push_back(Attribute::NOMINAL);
				data_vectors.emplace_back(std::vector<ScalarType>{});
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
				m_attributes.push_back(Attribute::DATE);
				data_vectors.emplace_back(std::vector<ScalarType>{});
			}
			else if (is_primitive_type(type))
			{
				type = to_lower(type);
				// numeric attributes
				if (type == "numeric")
				{
					m_attributes.push_back(Attribute::NUMERIC);
					data_vectors.emplace_back(std::vector<ScalarType>{});
				}
				else if (type == "integer")
				{
					m_attributes.push_back(Attribute::INTEGER);
					data_vectors.emplace_back(std::vector<ScalarType>{});
				}
				else if (type == "real")
				{
					m_attributes.push_back(Attribute::REAL);
					data_vectors.emplace_back(std::vector<ScalarType>{});
				}
				else if (type == "string")
				{
					// @ATTRIBUTE LCC    string
					m_attributes.push_back(Attribute::STRING);
					data_vectors.emplace_back(
					    std::vector<std::basic_string<CharType>>{});
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
			auto processed_name = trim(name, [](const auto& val) {
				return !std::isspace(val) && val != '\'' && val != '\"';
			});
			m_attribute_names.emplace_back(processed_name);
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

	// estimate the size of the @data section
	auto pos = m_stream->tellg();
	auto approx_data_line_count = std::count(
	    std::istreambuf_iterator<char>(*m_stream),
	    std::istreambuf_iterator<char>(), '\n');
	reserve_vector_memory(approx_data_line_count, data_vectors);
	m_stream->seekg(pos);

	std::vector<std::basic_string<CharType>> elems;
	elems.reserve(m_attributes.size());

	// read the @data section
	auto read_data = [this, &data_vectors, &elems]() {
		// it's a comment and can be skipped
		if (m_current_line.substr(0, 1) == m_comment_string)
			return;
		// it's the data string (i.e. @data"), does not provide information
		if (to_lower(m_current_line.substr(0, strlen(m_data_string))) ==
		    m_data_string)
			return;

		// assumes that until EOF we should expect comma delimited values
		elems.clear();
		split(m_current_line, ",", std::back_inserter(elems), "\'\"");
		if (elems.size() != m_attributes.size())
			SG_SERROR(
			    "Unexpected number of values on line %d, expected %d "
			    "values, but found %d.\n",
			    m_line_number, m_attributes.size(), elems.size())
		// only parse rows that do not contain missing values
		if (std::find(elems.begin(), elems.end(), m_missing_value_string) ==
		    elems.end())
		{
			auto nominal_pos = m_nominal_attributes.begin();
			auto date_pos = m_date_formats.begin();
			for (int i = 0; i < elems.size(); ++i)
			{
				Attribute type = m_attributes[i];
				switch (type)
				{
				case (Attribute::NUMERIC):
				case (Attribute::INTEGER):
				case (Attribute::REAL):
				{
					try
					{
						shogun::get<std::vector<ScalarType>>(data_vectors[i])
						    .push_back(buffer_to_type<ScalarType>(elems[i]));
					}
					catch (const std::invalid_argument&)
					{
						SG_SERROR(
						    "Failed to covert \"%s\" to numeric on line %d.\n",
						    elems[i].c_str(), m_line_number)
					}
				}
				break;
				case (Attribute::NOMINAL):
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
					ScalarType idx = std::distance(encoding.begin(), pos);
					shogun::get<std::vector<ScalarType>>(data_vectors[i])
					    .push_back(idx);
					++nominal_pos;
				}
				break;
				case (Attribute::DATE):
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
						shogun::get<std::vector<ScalarType>>(data_vectors[i])
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
				case (Attribute::STRING):
					shogun::get<std::vector<std::basic_string<CharType>>>(
					    data_vectors[i])
					    .emplace_back(elems[i]);
				}
			}
			++m_row_count;
		}
	};
	auto check_data = [&data_vectors]() {
		if (!data_vectors.empty())
		{
			auto feature_count = data_vectors.size();
			index_t row_count =
			    shogun::visit(VectorSizeVisitor{}, data_vectors[0]);
			for (int i = 1; i < feature_count; ++i)
			{
				REQUIRE(
				    shogun::visit(VectorSizeVisitor{}, data_vectors[i]) ==
				        row_count,
				    "All columns must have the same number of features!\n")
			}
		}
		else
			return false;
		return true;
	};
	process_chunk(read_data, check_data, true);

	// transform data into a feature object
	index_t row_count = shogun::visit(VectorSizeVisitor{}, data_vectors[0]);
	for (int i = 0; i < data_vectors.size(); ++i)
	{
		Attribute att = m_attributes[i];
		auto vec = data_vectors[i];
		switch (att)
		{
		case Attribute::NUMERIC:
		case Attribute::INTEGER:
		case Attribute::REAL:
		case Attribute::DATE:
		case Attribute::NOMINAL:
		{
			auto casted_vec = shogun::get<std::vector<ScalarType>>(vec);
			SGMatrix<ScalarType> mat(1, row_count);
			memcpy(
			    mat.matrix, casted_vec.data(),
			    casted_vec.size() * sizeof(ScalarType));
			m_features.emplace_back(new CDenseFeatures<ScalarType>(mat));
		}
		break;
		case Attribute::STRING:
		{
			auto casted_vec =
			    shogun::get<std::vector<std::basic_string<CharType>>>(vec);
			index_t max_string_length = 0;
			for (const auto& el : casted_vec)
			{
				if (max_string_length < el.size())
					max_string_length = el.size();
			}
			SGStringList<CharType> strings(row_count, max_string_length);
			for (int j = 0; j < row_count; ++j)
			{
				SGString<CharType> current(max_string_length);
				memcpy(
				    current.string, casted_vec[j].data(),
				    (casted_vec.size() + 1) * sizeof(CharType));
				strings.strings[j] = current;
			}
			m_features.emplace_back(
			    new CStringFeatures<CharType>(strings, EAlphabet::RAWBYTE));
		}
		}
	}
}

template <typename ScalarType>
void ARFFDeserializer::read_string_dispatcher()
{
	switch (m_string_primitive_type)
	{
	case EPrimitiveType::PT_UINT8:
	{
		read_helper<ScalarType, char>();
	}
	break;
	case EPrimitiveType::PT_UINT16:
	{
		SG_SNOTIMPLEMENTED
	}
	break;
	default:
		SG_SERROR("The provided type for string parsing is not valid!\n")
	}
}

void ARFFDeserializer::read()
{
	switch (m_primitive_type)
	{
	case EPrimitiveType::PT_INT8:
	{
		read_string_dispatcher<int8_t>();
	}
	break;
	case EPrimitiveType::PT_INT16:
	{
		read_string_dispatcher<int16_t>();
	}
	break;
	case EPrimitiveType::PT_INT32:
	{
		read_string_dispatcher<int32_t>();
	}
	break;
	case EPrimitiveType::PT_INT64:
	{
		read_string_dispatcher<int64_t>();
	}
	break;
	case EPrimitiveType::PT_FLOAT32:
	{
		read_string_dispatcher<float32_t>();
	}
	break;
	case EPrimitiveType::PT_FLOAT64:
	{
		read_string_dispatcher<float64_t>();
	}
	break;
	case EPrimitiveType::PT_FLOATMAX:
	{
		read_string_dispatcher<floatmax_t>();
	}
	break;
	default:
		SG_SERROR("The provided type for scalar parsing is not valid!\n")
	}
}

template <typename ScalarType, typename CharType>
void ARFFDeserializer::reserve_vector_memory(
    size_t line_count,
    std::vector<variant<
        std::vector<ScalarType>, std::vector<std::basic_string<CharType>>>>& v)
{
	VectorResizeVisitor visitor{line_count};
	for (auto& vec : v)
		shogun::visit(visitor, vec);
}