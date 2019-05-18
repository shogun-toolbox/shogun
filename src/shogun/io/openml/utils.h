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
		/**
		 * Checks the returned response from OpenML in JSON format
		 * @param doc the parsed OpenML JSON format response
		 */
		static void
		check_response(const rapidjson::Document& doc, const std::string& type)
		{
			if (SG_UNLIKELY(doc.HasMember("error")))
			{
				const rapidjson::Value& root = doc["error"];
				SG_SERROR(
				    "Server error %s: %s\n", root["code"].GetString(),
				    root["message"].GetString())
				return;
			}
			REQUIRE(
			    doc.HasMember(type.c_str()),
			    "Unexpected format of OpenML %s.\n", type.c_str());
		}

		/**
		 * Helper function to add JSON objects as string in map
		 * @param v a RapidJSON GenericValue, i.e. string
		 * @param param_dict the map to write to
		 * @param name the name of the key
		 */
		static SG_FORCED_INLINE void emplace_string_to_map(
		    const rapidjson::GenericValue<rapidjson::UTF8<char>>& v,
		    std::unordered_map<std::string, std::string>& param_dict,
		    const std::string& name, bool required = false)
		{
			if (v[name.c_str()].GetType() == rapidjson::Type::kStringType)
				param_dict.emplace(name, v[name.c_str()].GetString());
			else if (required)
				SG_SERROR(
				    "The field \"%s\" is expected to be a string!\n",
				    name.c_str())
			else
				param_dict.emplace(name, "");
		}

		/**
		 * Helper function to add JSON objects as string in map
		 * @param v a RapidJSON GenericObject, i.e. array
		 * @param param_dict the map to write to
		 * @param name the name of the key
		 */
		static SG_FORCED_INLINE void emplace_string_to_map(
		    const rapidjson::GenericObject<
		        true, rapidjson::GenericValue<rapidjson::UTF8<char>>>& v,
		    std::unordered_map<std::string, std::string>& param_dict,
		    const std::string& name)
		{
			if (v[name.c_str()].GetType() == rapidjson::Type::kStringType)
				param_dict.emplace(name, v[name.c_str()].GetString());
			else
				param_dict.emplace(name, "");
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
			SG_SERROR(
			    "\"%s\" is not a member of the given object", name.c_str())
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
