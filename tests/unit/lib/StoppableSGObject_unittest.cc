/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
*/

#include <functional>
#include <gtest/gtest.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/Signal.h>
#include <rxcpp/rx-lite.hpp>

using namespace shogun;
using namespace std;

class Mock_model : public CMachine
{
public:
	Mock_model() : m_check(0), m_i(0)
	{
		// Set up the custom callback
		function<bool()> callback = [this]()
		{
			// Stop if we did more than 5 steps
			if (m_i >= 5)
			{
				get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
				return true;
			}
			m_i++;
			return false;
		};

		// We then add the callback
		this->add_callback(callback);
	};

	int get_check()
	{
		return m_check;
	}

	virtual const char* get_name() const
	{
		return "Mock_model";
	}


protected:

	/** returns whether machine require labels for training */
	virtual bool train_require_labels() const { return false; }

	// Custom train machine
	virtual bool train_machine(CFeatures* data=NULL)
	{
		for (int k=0; k<10; k++)
		{
			COMPUTATION_CONTROLLERS
			m_check++;
		}
		return true;
	}
	int m_check;
	int m_i;
};



TEST(StoppableSGObject, custom_callback)
{
	Mock_model a;
	a.train();
	EXPECT_TRUE(a.get_check() == 5);
}

TEST(StoppableSGObject, custom_callback_by_user)
{
	int i=0;
	function<bool()> callback = [&i]()
	{
		if (i>=3) {
			get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
			return true;
		}
		i++;
		return false;
	};


	Mock_model a;
	a.add_callback(callback);
	a.train();
	EXPECT_TRUE(a.get_check() == 3);
}

