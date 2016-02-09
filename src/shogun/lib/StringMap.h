/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */
#ifndef STRINGMAP_H
#define STRINGMAP_H

#include <shogun/lib/config.h>
#include <shogun/lib/Map.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST
/** @brief The class is a customized map for the optimization framework. */
IGNORE_IN_CLASSLIST template<class T> class CStringMap: public CMap<std::string, T>
{
public:
	/** Custom constructor */
	CStringMap(int32_t size=41, int32_t reserved=128, bool tracable=true)
		:CMap<std::string, T>(size, reserved, tracable) {}

	/** Default destructor */
	virtual ~CStringMap() {}

	/** Return the name of the class
	 *
	 * @return name StringMap
	 */
	virtual const char* get_name() const { return "StringMap"; }

protected:
	/** Get the hash of a given string key
	 * @param key a given string key
	 * @return hash of the key
	 */
	virtual uint32_t get_hash_value(const std::string& key)
	{
		const char* k=key.c_str();
		return CHash::MurmurHash3((uint8_t*)(k), key.length(), 0xDEADBEEF);
	}

};

}

#endif /* STRINGMAP_H */
