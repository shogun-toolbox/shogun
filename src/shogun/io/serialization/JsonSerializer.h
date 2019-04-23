/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __JSON_SERIALIZER_H__
#define __JSON_SERIALIZER_H__

#include <shogun/io/serialization/Serializer.h>

namespace shogun
{
	namespace io
	{
		class JsonSerializer : public Serializer
		{
		public:
			JsonSerializer();
			~JsonSerializer() override;
			virtual void write(std::shared_ptr<SGObject> object) noexcept(false);

			virtual const char* get_name() const
			{
				return "JsonSerializer";
			}
		};
	}
}

#endif
