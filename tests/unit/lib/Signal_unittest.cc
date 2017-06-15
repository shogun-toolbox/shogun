#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>
#include <shogun/lib/Signal.h>

#include <csignal>

using namespace shogun;
using namespace rxcpp;

TEST(Signal, SIGINT_test)
{

	CSignal tmp;
	int on_next_v = 0;
	int on_complete_v = 0;
	auto sub = rxcpp::make_subscriber<int>(
	    [&on_next_v](int v) { on_next_v = 1; }, [&]() { on_complete_v = 1; });

	tmp.get_SIGINT_observable().subscribe(sub);

	tmp.handler(SIGINT);

	EXPECT_TRUE(on_next_v == 0);
	EXPECT_TRUE(on_complete_v == 1);
}

TEST(Signal, SIGURG_test)
{

	CSignal tmp;
	int on_next_v = 0;
	int on_complete_v = 0;
	auto sub = rxcpp::make_subscriber<int>(
	    [&](int v) { on_next_v++; }, [&]() { on_complete_v++; });

	tmp.get_SIGURG_observable().subscribe(sub);

	tmp.handler(SIGURG);

	EXPECT_TRUE(on_next_v == 1);
	EXPECT_TRUE(on_complete_v == 0);
}
