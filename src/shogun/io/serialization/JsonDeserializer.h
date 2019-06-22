/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __JSON_DESERIALIZER_H__
#define __JSON_DESERIALIZER_H__

#include <shogun/io/serialization/Deserializer.h>

namespace shogun
{
	namespace io
	{
		class JsonDeserializer : public Deserializer
		{
		public:
			JsonDeserializer();
			~JsonDeserializer() override;
			Some<CSGObject> read_object() override;
			void read(CSGObject* _this) override;
		};
	}
}

#endif
