/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */
#ifndef __BITSERY_DESERIALIZER_H__
#define __BITSERY_DESERIALIZER_H__

#include <shogun/io/serialization/Deserializer.h>

namespace shogun
{
	namespace io
	{
		class BitseryDeserializer : public Deserializer
		{
		public:
			BitseryDeserializer();
			~BitseryDeserializer() override;
			std::shared_ptr<SGObject> read_object() override;
			void read(std::shared_ptr<SGObject> _this) override;

			const char* get_name() const override
			{
				return "BitseryDeserializer";
			}
		};
	}
}

#endif
