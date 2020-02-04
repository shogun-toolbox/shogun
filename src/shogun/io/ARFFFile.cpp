/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/io/ARFFFile.h>
#include <shogun/lib/type_case.h>

#include <date/date.h>

using namespace shogun;
using namespace shogun::arff_detail;

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
	error(
	    "No conversion from {} to {}!\n", buffer.c_str(),
	    demangled_type<T>());
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
	std::vector<std::variant<
	    std::vector<ScalarType>, std::vector<std::basic_string<CharType>>>>
	    data_vectors;
	m_line_number = 0;
	m_row_count = 0;
	m_file_done = false;
	auto read_comment = [this]() {
		if (to_lower(m_current_line.substr(0, 1)) == m_comment_string)
			m_comments.push_back(m_current_line.substr(1, std::string::npos));
		else if (
		    to_lower(m_current_line.substr(0, m_relation_string.size())) ==
		    m_relation_string)
			m_state = true;
	};
	auto check_comment = []() { return true; };
	process_chunk(read_comment, check_comment, false);

	auto read_relation = [this]() {
		if (to_lower(m_current_line.substr(0, m_relation_string.size())) ==
		    m_relation_string)
			m_relation = remove_whitespace(
			    m_current_line.substr(m_relation_string.size()));
		else if (
		    to_lower(m_current_line.substr(0, m_attribute_string.size())) ==
		    m_attribute_string)
			m_state = true;
	};
	// a relation has to be defined
	auto check_relation = [this]() { return !m_relation.empty(); };
	process_chunk(read_relation, check_relation, true);

	// parse the @attributes section
	auto read_attributes = [this, &data_vectors]() {
		if (to_lower(m_current_line.substr(0, m_attribute_string.size())) ==
		    m_attribute_string)
		{
			std::string name, type;
			auto inner_string =
			    m_current_line.substr(m_attribute_string.size());
			left_trim(inner_string, [](const auto& val) {
				return !std::isspace(val);
			});
			auto it = inner_string.begin();
			if (is_part_of(*it, "\"\'"))
			{
				auto quote_type = *it;
				++it;
				auto begin = it;
				while (*it != quote_type && it != inner_string.end())
					++it;
				if (it == inner_string.end())
					error(
					    "Encountered unbalanced parenthesis in attribute "
					    "declaration on line {}: \"{}\"\n",
					    m_line_number, m_current_line);
				name = {begin, it};
				type = trim({std::next(it), inner_string.end()});
			}
			else
			{
				auto begin = it;
				while (!std::isspace(*it))
					++it;
				if (it == inner_string.end() && it != inner_string.end())
					error(
					    "Expected at least two elements in attribute "
					    "declaration on line {}: \"{}\"",
					    m_line_number, m_current_line);
				name = {begin, it};
				type = trim({std::next(it), inner_string.end()});
			}

			SG_DEBUG("name: {}\n", name);
			SG_DEBUG("type: {}\n", type);

			if (name.empty() || type.empty())
				error(
				    "Could not find the name and type on line {}: \"{}\".\n",
				    m_line_number, m_current_line);
			if (it == inner_string.end())
				error(
				    "Could not split attibute name and type on line {}: "
				    "\"{}\".\n",
				    m_line_number, m_current_line);

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
					error(
					    "Error parsing date on line {}: {}\n", m_line_number,
					    m_current_line);
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
					error(
					    "Unexpected attribute type identifier \"{}\" "
					    "on line {}: {}\n",
					    type, m_line_number, m_current_line);
			}
			else
				error(
				    "Unexpected format in @ATTRIBUTE on line {}: {}\n",
				    m_line_number, m_current_line);
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
		    to_lower(m_current_line.substr(0, m_data_string.size())) ==
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
		if (to_lower(m_current_line.substr(0, m_data_string.size())) ==
		    m_data_string)
			return;

		// assumes that until EOF we should expect comma delimited values
		elems.clear();
		split(m_current_line, ",", std::back_inserter(elems), "\'\"");
		if (elems.size() != m_attributes.size())
			error(
			    "Unexpected number of values on line {}, expected {} "
			    "values, but found {}.\n",
			    m_line_number, m_attributes.size(), elems.size());
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
						std::get<std::vector<ScalarType>>(data_vectors[i])
						    .push_back(buffer_to_type<ScalarType>(elems[i]));
					}
					catch (const std::invalid_argument&)
					{
						error(
						    "Failed to covert \"{}\" to numeric on line %d.\n",
						    elems[i], m_line_number);
					}
				}
				break;
				case (Attribute::NOMINAL):
				{
					if (nominal_pos == m_nominal_attributes.end())
						error(
						    "Unexpected nominal value \"{}\" on line {}\n",
						    elems[i].c_str(), m_line_number);
					auto encoding = (*nominal_pos).second;
					auto trimmed_el = trim(elems[i]);
					remove_char_inplace(trimmed_el, '\'');
					auto pos =
					    std::find(encoding.begin(), encoding.end(), trimmed_el);
					if (pos == encoding.end())
						error(
						    "Unexpected value \"{}\" on line %d\n",
						    trimmed_el, m_line_number);
					ScalarType idx = std::distance(encoding.begin(), pos);
					std::get<std::vector<ScalarType>>(data_vectors[i])
					    .push_back(idx);
					++nominal_pos;
				}
				break;
				case (Attribute::DATE):
				{
					date::sys_seconds t;
					std::istringstream ss(elems[i]);
					if (date_pos == m_date_formats.end())
						error(
						    "Unexpected date value \"{}\" on line {}.\n",
						    elems[i], m_line_number);
					ss >> date::parse(*date_pos, t);
					if (bool(ss))
					{
						auto value_timestamp = t.time_since_epoch().count();
						std::get<std::vector<ScalarType>>(data_vectors[i])
						    .push_back(value_timestamp);
					}
					else
						error(
						    "Error parsing date \"{}\" with date format \"{}\" "
						    "on line {}.\n",
						    elems[i], *date_pos,
						    m_line_number);
					++date_pos;
				}
				break;
				case (Attribute::STRING):
					std::get<std::vector<std::basic_string<CharType>>>(
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
			size_t row_count =
			    std::visit(VectorSizeVisitor{}, data_vectors[0]);
			for (int i = 1; i < feature_count; ++i)
			{
				require(
				    std::visit(VectorSizeVisitor{}, data_vectors[i]) ==
				        row_count,
				    "All columns must have the same number of features!\n");
			}
		}
		else
			return false;
		return true;
	};
	process_chunk(read_data, check_data, true);

	// transform data into a feature object
	index_t row_count = std::visit(VectorSizeVisitor{}, data_vectors[0]);
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
			auto casted_vec = std::get<std::vector<ScalarType>>(vec);
			SGMatrix<ScalarType> mat(1, row_count);
			memcpy(
			    mat.matrix, casted_vec.data(),
			    casted_vec.size() * sizeof(ScalarType));
			m_features.push_back(std::make_shared<DenseFeatures<ScalarType>>(mat));
		}
		break;
		case Attribute::STRING:
		{
			auto casted_vec =
			    std::get<std::vector<std::basic_string<CharType>>>(vec);
			index_t max_string_length = 0;
			for (const auto& el : casted_vec)
			{
				if (max_string_length < el.size())
					max_string_length = el.size();
			}
			std::vector<SGVector<CharType>> strings(row_count, max_string_length);
			for (int j = 0; j < row_count; ++j)
			{
				SGVector<CharType> current(max_string_length);
				memcpy(
				    current.vector, casted_vec[j].data(),
				    (casted_vec.size() + 1) * sizeof(CharType));
				strings[j] = current;
			}
			m_features.push_back(
			    std::make_shared<StringFeatures<CharType>>(strings, EAlphabet::RAWBYTE));
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
		error("16-bit wide string conversion not available.");
	}
	break;
	default:
		error("The provided type for string parsing is not valid!\n");
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
		error("The provided type for scalar parsing is not valid!\n");
	}
}

template <typename ScalarType, typename CharType>
void ARFFDeserializer::reserve_vector_memory(
    size_t line_count,
    std::vector<std::variant<
        std::vector<ScalarType>, std::vector<std::basic_string<CharType>>>>& v)
{
	VectorResizeVisitor visitor{line_count};
	for (auto& vec : v)
		std::visit(visitor, vec);
}

/**
 * Very type unsafe, but no UB!
 * @param obj
 * @return
 */
std::vector<std::string> features_to_string(const std::shared_ptr<SGObject>& obj, Attribute att)
{
	std::vector<std::string> result_string;
	switch (att)
	{
	case Attribute::NUMERIC:
	case Attribute::REAL:
	{
		auto mat_to_string = [&result_string](const auto& mat) {
			result_string.reserve(mat.size());
			for (int i = 0; i < mat.size(); ++i)
			{
				result_string.push_back(std::to_string(mat[i]));
			}
		};

		for (const auto& param : obj->get_params())
			if (param.first == "feature_matrix")
			{
				sg_any_dispatch(
				    param.second->get_value(), sg_matrix_typemap,
				    shogun::None{}, shogun::None{}, mat_to_string);
				return result_string;
			}
	}
	break;
	case Attribute::INTEGER:
	{
		auto mat_to_string = [&result_string](const auto& mat) {
			result_string.reserve(mat.size());
			for (int i = 0; i < mat.size(); ++i)
			{
				result_string.push_back(
				    std::to_string(static_cast<int64_t>(mat[i])));
			}
		};

		for (const auto& param : obj->get_params())
			if (param.first == "feature_matrix")
			{
				sg_any_dispatch(
				    param.second->get_value(), sg_matrix_typemap,
				    shogun::None{}, shogun::None{}, mat_to_string);
				return result_string;
			}
	}
	break;
	default:
		error("Unsupported type: {}\n", static_cast<int>(att));
	}
	error("The provided feature object does not have a feature matrix!\n");
	return std::vector<std::string>{};
}

std::vector<std::string> features_to_string(
    const std::shared_ptr<SGObject>& obj, const std::vector<std::string>& nominal_values)
{
	std::vector<std::string> result_string;
	auto mat_to_string = [&result_string, &nominal_values](const auto& mat) {
		result_string.reserve(mat.size());
		for (int i = 0; i < mat.size(); ++i)
		{
			result_string.emplace_back(
			    "\"" + nominal_values[static_cast<size_t>(mat[i])] + "\"");
		}
	};

	for (const auto& param : obj->get_params())
		if (param.first == "feature_matrix")
		{
			sg_any_dispatch(
			    param.second->get_value(), sg_matrix_typemap, shogun::None{},
			    shogun::None{}, mat_to_string);
			return result_string;
		}
	error("The provided feature object does not have a feature matrix!\n");
	return std::vector<std::string>{};
}

std::unique_ptr<std::ostringstream> ARFFSerializer::write()
{
	auto ss = std::make_unique<std::ostringstream>();

	// @relation
	*ss << ARFFDeserializer::m_relation_string << " " << m_name << "\n\n";

	// @attribute
	for (const auto& att : m_attributes)
	{
		switch (att.second)
		{
		case Attribute::NUMERIC:
			*ss << ARFFDeserializer::m_attribute_string << " " << att.first
			   << " numeric\n";
			break;
		case Attribute::INTEGER:
			*ss << ARFFDeserializer::m_attribute_string << " " << att.first
			   << " integer\n";
			break;
		case Attribute::REAL:
			*ss << ARFFDeserializer::m_attribute_string << " " << att.first
			   << " real\n";
			break;
		case Attribute::STRING:
			*ss << ARFFDeserializer::m_attribute_string << " " << att.first
			   << " string\n";
			break;
		case Attribute::DATE:
			error("C++ to Java date format conversion is not implement!");
			break;
		case Attribute::NOMINAL:
		{
			*ss << ARFFDeserializer::m_attribute_string << " " << att.first
			   << " ";
			auto nominal_values_vector = m_nominal_mapping.at(att.first);
			std::string nominal_values_string = std::accumulate(
			    nominal_values_vector.begin(), nominal_values_vector.end(),
			    "{\"" + nominal_values_vector[0] + "\"",
			    [](std::string& lhs, const std::string& rhs) {
				    return lhs += ",\"" + rhs + "\"";
			    });
			nominal_values_string.append("}\n");
			*ss << nominal_values_string;
		}
		}
	}

	// @data
	*ss << "\n" << ARFFDeserializer::m_data_string << "\n\n";

	auto num_vectors = m_feature_list.back()->as<Features>()->get_num_vectors();
	std::vector<std::vector<std::string>> result;
	auto att_iter = m_attributes.begin();

	for (const auto& feature: m_feature_list)
	{
		auto n_i = feature->as<Features>()->get_num_vectors();
		require(
		    n_i == num_vectors,
		    "Expected all features to have the same number of examples!\n");

		switch (att_iter->second)
		{
		case Attribute::NUMERIC:
		case Attribute::REAL:
		case Attribute::INTEGER:
			result.push_back(features_to_string(feature, att_iter->second));
			break;
		case Attribute::NOMINAL:
			result.push_back(
			    features_to_string(feature, m_nominal_mapping.at(att_iter->first)));
			break;
		case Attribute::DATE:
		case Attribute::STRING:
			error("Writing out strings and dates has not been implemented!");
		}
		++att_iter;
	}

	std::vector<std::string> result_rows(num_vectors);

	for (size_t col = 0; col != result.size(); ++col)
	{
		if (col != result.size() - 1)
			for (auto row = 0; row != num_vectors; ++row)
				result_rows[row].append(result[col][row] + ",");
		else
			for (auto row = 0; row != num_vectors; ++row)
				result_rows[row].append(result[col][row] + "\n");
	}

	for (const auto& row : result_rows)
		*ss << row;

	return ss;
}

void ARFFSerializer::write(const std::string& filename)
{
	auto result = write();
	std::ofstream myfile;
	myfile.open(filename);
	myfile << result->str();
	myfile.close();
}
