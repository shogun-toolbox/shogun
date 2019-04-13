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
		class CJsonDeserializer : public CDeserializer
		{
		public:
			CJsonDeserializer();
			~CJsonDeserializer() override;
			Some<CSGObject> read_object() override;
			void read(CSGObject* _this) override;
			virtual const char* get_name() const
			{
				return "JsonDeserializer";
			}
		};
	}
}

#endif
