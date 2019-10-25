/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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
 *
 */


/* Remove C Prefix */
%shared_ptr(shogun::Minimizer)
%shared_ptr(shogun::LBFGSMinimizer)
%shared_ptr(shogun::FirstOrderMinimizer)
%shared_ptr(shogun::SingleLaplaceNewtonOptimizer)
%shared_ptr(shogun::SingleFITCLaplaceNewtonOptimizer)

#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
%shared_ptr(shogun::NLOPTMinimizer)
#endif //HAVE_NLOPT
#endif //USE_GPL_SHOGUN



/* These functions return new Objects */

/* Include Class Headers to make them visible from within the target language */
%include <shogun/optimization/Minimizer.h>
%include <shogun/optimization/FirstOrderMinimizer.h>
%include <shogun/optimization/lbfgs/lbfgscommon.h>
%include <shogun/optimization/lbfgs/LBFGSMinimizer.h>
#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
%include <shogun/optimization/nloptcommon.h>
%include <shogun/optimization/NLOPTMinimizer.h>
#endif //HAVE_NLOPT
#endif //USE_GPL_SHOGUN
