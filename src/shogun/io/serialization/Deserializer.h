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
		class Deserializer : public SGObject
		{
		public:
			Deserializer();
			virtual ~Deserializer();
			virtual void attach(std::shared_ptr<io::InputStream> stream);
			virtual std::shared_ptr<SGObject> read_object() = 0;
			virtual void read(std::shared_ptr<SGObject> _this) = 0;
			std::shared_ptr<io::InputStream> stream() const;

		private:
			std::shared_ptr<io::InputStream> m_stream;
		};

		std::shared_ptr<SGObject> deserialize(const std::string& _path, const std::shared_ptr<Deserializer>& _deser);
		void pre_deserialize(const std::shared_ptr<SGObject>& obj) noexcept(false);
		void post_deserialize(const std::shared_ptr<SGObject>& obj) noexcept(false);

	}
}

#endif
