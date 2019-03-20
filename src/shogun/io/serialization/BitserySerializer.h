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
		class CBitserySerializer : public CSerializer
		{
		public:
			CBitserySerializer();
			~CBitserySerializer() override;
			void write(Some<CSGObject> object) override;

			const char* get_name() const override
			{
				return "BitserySerializer";
			}
		};
	}
}

#endif
