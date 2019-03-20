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
		class CBitseryDeserializer : public CDeserializer
		{
		public:
			CBitseryDeserializer();
			virtual ~CBitseryDeserializer();
			virtual Some<CSGObject> read();

			virtual const char* get_name() const
			{
				return "BitseryDeserializer";
			}
		};
	}
}

#endif
