/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENML_UTILS_H
#define SHOGUN_OPENML_UTILS_H

#include <shogun/io/SGIO.h>

#include <rapidjson/document.h>

namespace shogun
{
	namespace openml_detail
	{
		enum class BACKEND_FORMAT
		{
			JSON = 0,
			XML = 1,
		};

		/**
		 * Checks the returned response from OpenML in JSON format
		 * @param doc the parsed OpenML JSON format response
		 */
		template <
		    BACKEND_FORMAT FormatT,
		    typename std::enable_if_t<FormatT == BACKEND_FORMAT::JSON>* =
		        nullptr>
		const rapidjson::Value&
		check_response(const std::string& val, const std::string& root_name)
		{
			rapidjson::Document doc;
			doc.Parse(val.c_str());

			if (SG_UNLIKELY(doc.HasMember("error")))
			{
				const rapidjson::Value& root = doc["error"];
				SG_SERROR(
				    "Server error %s: %s\n", root["code"].GetString(),
				    root["message"].GetString())
			}
			REQUIRE(
			    doc.HasMember(root_name.c_str()),
			    "Unexpected format of OpenML %s.\n", root_name.c_str());

			return doc[root_name.c_str()];
		}

		/**
		 * Checks the returned response from OpenML in XML format
		 * @param doc the parsed OpenML XML format response
		 */
		template <
		    BACKEND_FORMAT FormatT,
		    typename std::enable_if_t<FormatT == BACKEND_FORMAT::XML>* =
		        nullptr>
		void check_response(const std::string& val, const std::string& type)
		{
			SG_SNOTIMPLEMENTED
		}

		template <typename T>
		static SG_FORCED_INLINE void add_string_to_struct(
		    const rapidjson::GenericObject<
		        true, rapidjson::GenericValue<rapidjson::UTF8<char>>>& v,
		    const std::string& name, T& custom_struct)
		{
			if (v[name.c_str()].GetType() == rapidjson::Type::kStringType)
				custom_struct = v[name.c_str()].GetString();
		}

		template <typename T>
		static SG_FORCED_INLINE void add_string_to_struct(
		    const rapidjson::GenericValue<rapidjson::UTF8<char>>& v,
		    const std::string& name, T& custom_struct)
		{
			if (v[name.c_str()].GetType() == rapidjson::Type::kStringType)
				custom_struct = v[name.c_str()].GetString();
		}

		template <typename T>
		SG_FORCED_INLINE T must_return(
		    const std::string& name,
			const rapidjson::GenericValue<rapidjson::UTF8<char>>& v)
		{
			SG_SNOTIMPLEMENTED
		}

		template <>
		SG_FORCED_INLINE std::string must_return<std::string>(
				const std::string& name,
				const rapidjson::GenericValue<rapidjson::UTF8<char>>& v)
		{
			if (v.HasMember(name.c_str()) && v[name.c_str()].IsString())
				return v[name.c_str()].GetString();
			if (v.HasMember(name.c_str()) && !v[name.c_str()].IsString())
				SG_SERROR(
					"Found member \"%s\" but it is not a string", name.c_str())
			if (!v.HasMember(name.c_str()))
				SG_SERROR(
					"\"%s\" is not a member of the given object", name.c_str())
			return nullptr;
		}


		template <typename T>
		SG_FORCED_INLINE T return_if_possible(
		    const std::string& name,
		    const rapidjson::GenericObject<
		        true, rapidjson::GenericValue<rapidjson::UTF8<char>>>& v)
		{
			SG_SNOTIMPLEMENTED
		}

		template <>
		SG_FORCED_INLINE std::string return_if_possible<std::string>(
		    const std::string& name,
		    const rapidjson::GenericObject<
		        true, rapidjson::GenericValue<rapidjson::UTF8<char>>>& v)
		{
			if (v.HasMember(name.c_str()) && v[name.c_str()].IsString())
				return v[name.c_str()].GetString();
			if (v.HasMember(name.c_str()) && !v[name.c_str()].IsString())
				SG_SERROR(
				    "Found member \"%s\" but it is not a string", name.c_str())
			if (!v.HasMember(name.c_str()))
				return "";
			return nullptr;
		}

		template <>
		SG_FORCED_INLINE std::vector<std::string>
		return_if_possible<std::vector<std::string>>(
		    const std::string& name,
		    const rapidjson::GenericObject<
		        true, rapidjson::GenericValue<rapidjson::UTF8<char>>>& v)
		{
			std::vector<std::string> result;
			if (!v.HasMember(name.c_str()))
				SG_SERROR(
				    "\"%s\" is not a member of the given object", name.c_str())
			if (v[name.c_str()].IsString())
			{
				result.emplace_back(v[name.c_str()].GetString());
			}
			if (v[name.c_str()].IsArray())
			{
				for (const auto& val : v[name.c_str()].GetArray())
				{
					if (val.IsString())
						result.emplace_back(val.GetString());
					else
						SG_SERROR(
						    "Found non string member in \"%s\".\n",
						    name.c_str())
				}
			}
			return result;
		}
	} // namespace openml_detail
} // namespace shogun
#endif // SHOGUN_OPENML_UTILS_H
