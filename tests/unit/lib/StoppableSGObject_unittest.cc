/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
*/

#include <functional>
#include <gtest/gtest.h>
#include <rxcpp/rx-lite.hpp>
#include <shogun/lib/Signal.h>
#include <shogun/machine/Machine.h>

using namespace shogun;
using namespace std;

/**
 * Mock model to show the use of the callback.
 */
#if 0
class MockModel : public Machine
{
public:
	MockModel() : m_check(0), m_last_iteration(0)
	{
		// Set up the custom callback
		function<bool()> callback = [std::shared_ptr<> if we did more than 5 steps
			if (m_last_iteration >= 5)
			{
				get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
				return true;
			}
			m_last_iteration++;
			return false;
		};

		this->set_callback(callback);
	};

	int get_check()
	{
		return m_check;
	}

	virtual const char* get_name() const
	{
		return "MockModel";
	}

protected:
	virtual bool train_require_labels() const
	{
		return false;
	}

	/* Custom train machine */
	virtual bool train_machine(std::shared_ptr<Features> data = NULL)
	{
		for (int num_iterations_train = 0; num_iterations_train < 10;
		     num_iterations_train++)
		{
			COMPUTATION_CONTROLLERS
			m_check++;
		}
		return true;
	}

	/* Control variable, used to check that we stopped the training at the
	 * exact number of iterations (it will be equal to m_last_iteration)*/
	int m_check;

	/* Addition control variable that is incremented each time by the
	 * callback.*/
	int m_last_iteration;
};

TEST(StoppableSGObject, empty_callback)
{
	MockModel a;
	a.set_callback(nullptr);
	a.train();
	EXPECT_TRUE(a.get_check() == 10);
}

TEST(StoppableSGObject, default_callback)
{
	MockModel a;
	a.train();
	EXPECT_TRUE(a.get_check() == 5);
}

TEST(StoppableSGObject, custom_callback_by_user)
{
	int i = 0;
	function<bool()> callback = [&i]() {
		if (i >= 3)
		{
			get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
			return true;
		}
		i++;
		return false;
	};

	MockModel a;
	a.set_callback(callback);
	a.train();
	EXPECT_TRUE(a.get_check() == 3);
}
#endif
