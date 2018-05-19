/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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
 */

#ifndef _EMBASE_H__
#define _EMBASE_H__

#include <shogun/lib/config.h>
#include <shogun/base/progress.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief This is the base class for Expectation Maximization (EM). EM for various purposes can be derived from
 * this base class. This is a template class having a template member called data which can be used to store all
 * parameters used and results calculated by the expectation and maximization steps of EM.
 */
template <class T> class CEMBase : public CSGObject
{
	public:
		/** constructor */
		CEMBase() : CSGObject() { };

		/** destructor */
		virtual ~CEMBase() { };

		/** returns the name of the class */
		virtual const char* get_name() const { return "EMBase"; }

		/** expectation step
		 *
		 * @return updated value log_likelihood
		 */
		virtual float64_t expectation_step()=0;

		/** maximization step */
		virtual void maximization_step()=0;

		/** Expectation Maximization algorithm - runs expectation step and maximization step repeatedly as long as
		 * max number of iterations is not reached or convergence does not take place.
		 *
		 * @param max_iters max number of iterations of EM
		 * @param epsilon convergence tolerance
		 * @return whether convergence is acheived
		 */
		bool iterate_em(int32_t max_iters=10000, float64_t epsilon=1e-8)
		{
			float64_t log_likelihood_cur=0;
			float64_t log_likelihood_prev=0;
			int32_t i=0;
			bool converge=false;
			auto pb = progress(range(max_iters));
			while (i<max_iters)
			{
				log_likelihood_prev=log_likelihood_cur;
				log_likelihood_cur=expectation_step();

				if ((i>0)&&(log_likelihood_cur-log_likelihood_prev)<epsilon)
				{
					converge=true;
					break;
				}

				maximization_step();
				i++;
				pb.print_progress();
			}
			pb.complete();
			return converge;
		}

	public:
		/** data */
		T data;

};
} /* shogun */
#endif /* _EMBASE_H__ */
