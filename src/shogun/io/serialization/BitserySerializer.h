/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */
#ifndef __BITSERY_SERIALIZER_H__
#define __BITSERY_SERIALIZER_H__

#include <shogun/io/serialization/Serializer.h>

namespace shogun
{
	namespace io
	{
		class BitserySerializer : public Serializer
		{
		public:
			BitserySerializer();
			~BitserySerializer() override;
			void write(const std::shared_ptr<SGObject>& object) override;

			const char* get_name() const override
			{
				return "BitserySerializer";
			}
		};
	}
}

#endif
