/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
*/
#include <gtest/gtest.h>

#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#include <shogun/lib/observers/ParameterObserverScalar.h>
#include <vector>

std::vector<std::string> test_params = {"a", "b", "c", "d"};

using namespace shogun;

TEST(ParameterObserverScalar, filter_empty)
{
	ParameterObserverScalar tmp;
	EXPECT_TRUE(tmp.filter("a"));
}

TEST(ParameterObserverScalar, filter_found)
{
	ParameterObserverScalar tmp{test_params};
	EXPECT_TRUE(tmp.filter("a"));
	EXPECT_TRUE(tmp.filter("b"));
	EXPECT_TRUE(tmp.filter("c"));
	EXPECT_TRUE(tmp.filter("d"));
}

TEST(ParameterObserverScalar, filter_not_found)
{
	ParameterObserverScalar tmp{test_params};
	EXPECT_FALSE(tmp.filter("k"));
}

#endif // HAVE_TFLOGGER
