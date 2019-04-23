/*
* This software is distributed under BSD 3-clause license (see LICENSE file).
*
* Authors: Shubham Shukla
*/

#include <rxcpp/rx-lite.hpp>
#include <shogun/base/init.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/StoppableSGObject.h>

using namespace shogun;

StoppableSGObject::StoppableSGObject() : SGObject()
{
	m_cancel_computation = false;
	m_pause_computation_flag = false;

	m_callback = nullptr;
};

StoppableSGObject::~StoppableSGObject(){};

rxcpp::subscription StoppableSGObject::connect_to_signal_handler()
{
	// Subscribe this algorithm to the signal handler
	auto subscriber = rxcpp::make_subscriber<int>(
	    [this](int i) {
		    if (i == SG_PAUSE_COMP)
			    this->on_pause();
		    else
			    this->on_next();
		},
	    [this]() { this->on_complete(); });
	return get_global_signal()->get_observable()->subscribe(subscriber);
}

void StoppableSGObject::set_callback(std::function<bool()> callback)
{
	m_callback = callback;
}

void StoppableSGObject::reset_computation_variables()
{
	m_cancel_computation = false;
	m_pause_computation_flag = false;
}

void StoppableSGObject::on_next()
{
	m_cancel_computation.store(true);
	on_next_impl();
}

void StoppableSGObject::on_pause()
{
	m_pause_computation_flag.store(true);
	on_pause_impl();
	resume_computation();
}

void StoppableSGObject::on_complete()
{
	on_complete_impl();
}
void StoppableSGObject::on_next_impl()
{
}
void StoppableSGObject::on_pause_impl()
{
}
void StoppableSGObject::on_complete_impl()
{
}
