/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
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
#include <shogun/lib/StringMap.h>
#include <gtest/gtest.h>
#include <shogun/lib/config.h>

using namespace shogun;

TEST(StringMap, test1)
{
	CStringMap<SGVector<float64_t> > smap;

	const char* name="stringmapunittests";
	std::string k1=std::string(name);

	SGVector<float64_t> v(2);
	v.set_const(0);
	smap.add(k1,v);
	EXPECT_TRUE(smap.contains(k1));

	std::string k2=k1;
	EXPECT_FALSE(&k1 == &k2);

	EXPECT_TRUE(smap.contains(k2));
}

TEST(StringMap, test2)
{
	CStringMap<SGVector<float64_t> > smap;

	const char* name1="stringmapunittests1";
	std::string k1=std::string(name1);

	SGVector<float64_t> v(2);
	v.set_const(0);
	smap.add(k1,v);
	EXPECT_TRUE(smap.contains(k1));

	const char* name2="stringmapunittests2";
	std::string k2=std::string(name2);
	EXPECT_FALSE(smap.contains(k2));

	const char* name3="stringmapunittests";
	std::string k3=std::string(name3);
	EXPECT_FALSE(smap.contains(k3));

	const char* name4="stringmapunittests11";
	std::string k4=std::string(name4);
	EXPECT_FALSE(smap.contains(k4));
}
