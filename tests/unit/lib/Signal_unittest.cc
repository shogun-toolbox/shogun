/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */
#include <gtest/gtest.h>
#include <shogun/lib/Signal.h>

#include <csignal>
#include <rxcpp/rx-lite.hpp>

using namespace shogun;
using namespace rxcpp;

class SignalFixture : public ::testing::Test
{
protected:
	virtual void SetUp()
	{

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
		    [&]() {
			    on_complete_v++;
			    fprintf(stderr, "Application Killed");
			});

		tmp.get_observable()->subscribe(sub);
	}

	CSignal tmp;
	int on_next_v;
	int on_complete_v;
};

TEST_F(SignalFixture, return_to_prompt_test)
{
	EXPECT_EXIT(
	    { std::raise(SIGINT); }, ::testing::ExitedWithCode(0),
	    "Application Killed");
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
