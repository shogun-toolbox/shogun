
#include <regex>
#include <shogun/io/fs/Path.h>
#include <shogun/lib/exception/ShogunException.h>

using namespace std;

namespace shogun
{
	namespace io
	{
		namespace detail
		{
			string join_path(std::initializer_list<string_view> paths)
			{
				string result;
				for (string_view path: paths)
				{
					if (path.empty()) continue;

					if (result.empty())
					{
						result = string(path);
						continue;
					}

					if (result[result.size() - 1] == '/')
					{
						if (is_absolute_path(path))
						{
							auto p = path.substr(1);
							result.append(p.data(), p.size());
						}
						else
							result.append(path.data(), path.size());
					}
					else
					{
						if (is_absolute_path(path))
							result.append(path.data(), path.size());
						else
							result.append("/").append(path.data(), path.size());
					}
				}
				return result;
			}

			string_view submatch(const cmatch& match, int32_t idx)
			{
				return string_view(match[idx].first, match[idx].length());
			}
		}

		bool is_absolute_path(string_view path)
		{
		  return !path.empty() && path[0] == '/';
		}

		void parse_uri(
			string_view uri, string_view* scheme,
			string_view* host, string_view* path)
		{
			static const regex uri_regex(
				R"(^(([^:\/?#]+):)?(//([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?)",
    			std::regex::extended);

			cmatch match;
			if (!regex_match(uri.data(), match, uri_regex))
				throw ShogunException("Provided string is not an URI!");

			*scheme = detail::submatch(match, 2);
			*host = detail::submatch(match, 4);
			*path = detail::submatch(match, 5);
		}
	}
}
