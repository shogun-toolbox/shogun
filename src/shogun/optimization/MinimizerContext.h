/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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

#ifndef MINIMIZERCONTEXT_H
#define MINIMIZERCONTEXT_H
#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
class CMinimizerContext: public CSGObject
{
public:
	/*  Constructor */
	CMinimizerContext()
		:CSGObject()
	{
		init();
	}

	/*  Destructor */
	virtual ~CMinimizerContext()
	{
		SG_UNREF(m_additional_context);
	}

	/** Returns the name of the inference method
	 *
	 * @return name MinimizerContext
	 */
	virtual const char* get_name() const {return "MinimizerContext";}

	/*  Used in gradient updater class */
	SGVector<float64_t> m_corrected_direction;

	/*  Used in learn rate class */
	int32_t m_learning_rate_count;

	/*  Store additional information */
	CSGObject* m_additional_context;
private:
	/*  Init */
	void init()
	{
		m_learning_rate_count=0;
		m_additional_context=NULL;
		m_corrected_direction=SGVector<float64_t>();
		SG_ADD(&m_corrected_direction, "corrected_direction", "corrected_direction", MS_NOT_AVAILABLE);
		SG_ADD(&m_learning_rate_count, "learning_rate_count", "learning_rate_count", MS_NOT_AVAILABLE);
		SG_ADD(&m_additional_context, "additional_context", "additional_context", MS_NOT_AVAILABLE);
	}
	
};
}

#endif
