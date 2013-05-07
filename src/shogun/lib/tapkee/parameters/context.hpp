/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_CONTEXT_H_
#define TAPKEE_CONTEXT_H_

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{

class Context
{
public:

	Context(void (*progress)(double), bool (*cancel)()) :
		progress_function(progress), cancel_function(cancel)
	{
	}

	inline void report_progress(double x) const
	{
		if (progress_function)
			progress_function(x);
	}

	inline bool is_cancelled() const
	{
		if (cancel_function)
			return cancel_function();
		return false;
	}

private:
	void (*progress_function)(double);
	bool (*cancel_function)();
};

}
}

#endif
