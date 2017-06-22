#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>
#include <shogun/lib/Signal.h>

#include <csignal>

using namespace shogun;
using namespace rxcpp;

TEST(Signal, return_to_prompt_test)
{
	CSignal tmp;
	tmp.enable_handler();
	int on_next_v = 0;
	int on_complete_v = 0;
	auto sub = rxcpp::make_subscriber<int>(
	    [&on_next_v](int v) { on_next_v = 1; }, [&]() { on_complete_v = 1; });

	tmp.get_observable().subscribe(sub);
	tmp.get_subscriber().on_completed();

	EXPECT_TRUE(on_complete_v == 1);
	EXPECT_TRUE(on_next_v == 0);
	CSignal::reset_handler();
}

TEST(Signal, prematurely_stop_computation_test)
{

	CSignal tmp;
	tmp.enable_handler();
	int on_next_v = 0;
	int on_complete_v = 0;
	auto sub = rxcpp::make_subscriber<int>(
	    [&](int v) { on_next_v++; }, [&]() { on_complete_v++; });

	tmp.get_observable().subscribe(sub);
	tmp.get_subscriber().on_next(SG_BLOCK_COMP);

	EXPECT_TRUE(on_next_v == 1);
	EXPECT_TRUE(on_complete_v == 0);
	CSignal::reset_handler();
}

TEST(Signal, pause_computation_test)
{

	CSignal tmp;
	tmp.enable_handler();
	int on_next_v = 0;
	int on_complete_v = 0;
	auto sub = rxcpp::make_subscriber<int>(
	    [&](int v) {
		    if (v == SG_PAUSE_COMP)
			    on_next_v += 2;
		    else
			    on_next_v++;
		},
	    [&]() { on_complete_v++; });

	tmp.get_observable().subscribe(sub);
	tmp.get_subscriber().on_next(SG_PAUSE_COMP);

	EXPECT_TRUE(on_next_v == 2);
	EXPECT_TRUE(on_complete_v == 0);
	CSignal::reset_handler();
}
