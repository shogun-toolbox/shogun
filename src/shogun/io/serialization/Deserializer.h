/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __DESERIALIZER_H__
#define __DESERIALIZER_H__

#include <shogun/base/SGObject.h>
#include <shogun/io/stream/InputStream.h>

namespace shogun
{
	namespace io
	{
		class CDeserializer : public CSGObject
		{
		public:
			CDeserializer();
			virtual ~CDeserializer();
			virtual void attach(Some<CInputStream> stream);
			virtual Some<CSGObject> read_object() = 0;
			virtual void read(CSGObject* _this) = 0;
			Some<CInputStream> stream() const;

		private:
			Some<CInputStream> m_stream;
		};

		CSGObject* deserialize(const std::string& _path, CDeserializer* _deser);
		void pre_deserialize(CSGObject* obj) noexcept(false);
		void post_deserialize(CSGObject* obj) noexcept(false);

	}
}

#endif
