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
		class Serializer : public SGObject
		{
		public:
			Serializer();
			virtual ~Serializer();
			virtual void attach(std::shared_ptr<OutputStream> stream);
			virtual void write(std::shared_ptr<SGObject> object) noexcept(false) = 0;
			std::shared_ptr<OutputStream> stream() const;

		private:
			std::shared_ptr<OutputStream> m_stream;
		};

		void serialize(const std::string& _path, std::shared_ptr<SGObject> _obj, std::shared_ptr<Serializer> _serializer);
		void pre_serialize(std::shared_ptr<SGObject> obj) noexcept(false);
		void post_serialize(std::shared_ptr<SGObject> obj) noexcept(false);
	}
}

#endif
