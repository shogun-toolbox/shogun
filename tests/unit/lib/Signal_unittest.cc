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
#include <shogun/lib/Signal.h>

#include <csignal>
#include <rxcpp/rx-lite.hpp>

using namespace shogun;
using namespace rxcpp;

class SignalFixture : public ::testing::Test {
protected:
	virtual void SetUp() {

		CSignal::reset_handler();
		on_next_v = 0;
		on_complete_v = 0;
		std::signal(SIGINT, tmp.handler);
		std::signal(SIGQUIT, tmp.handler);
		std::signal(SIGTSTP, tmp.handler);
		tmp.interactive(false);
		auto sub = rxcpp::make_subscriber<int>(
				[&](int v) {
					if (v == SG_PAUSE_COMP)
						on_next_v += 2;
					else
						on_next_v++;
				},
				[&]() { on_complete_v++; fprintf(stderr,"Application Killed");});

		tmp.get_observable()->subscribe(sub);
	}

	CSignal tmp;
	int on_next_v;
	int on_complete_v;
};

TEST_F(SignalFixture, return_to_prompt_test)
{
	EXPECT_EXIT({std::raise(SIGINT);}, ::testing::ExitedWithCode(0), "Application Killed");
}

TEST_F(SignalFixture, prematurely_stop_computation_test)
{
	std::raise(SIGQUIT);
	EXPECT_TRUE(on_next_v == 1);
	EXPECT_TRUE(on_complete_v == 0);
}

TEST_F(SignalFixture, pause_computation_test)
{
	std::raise(SIGTSTP);
	EXPECT_TRUE(on_next_v == 2);
	EXPECT_TRUE(on_complete_v == 0);
}
