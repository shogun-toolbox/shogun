/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __SERIALIZER_H__
#define __SERIALIZER_H__

#include <shogun/base/SGObject.h>
#include <shogun/io/stream/OutputStream.h>

namespace shogun
{
	namespace io
	{
		class CSerializer : public CSGObject
		{
		public:
			CSerializer();
			virtual ~CSerializer();
			virtual void attach(Some<COutputStream> stream);
			virtual void write(Some<CSGObject> object) noexcept(false) = 0;
			Some<COutputStream> stream() const;

		private:
			Some<COutputStream> m_stream;
		};

		void serialize(const std::string& _path, CSGObject* _obj, CSerializer* _serializer);
		void pre_serialize(CSGObject* obj) noexcept(false);
		void post_serialize(CSGObject* obj) noexcept(false);
	}
}

#endif
