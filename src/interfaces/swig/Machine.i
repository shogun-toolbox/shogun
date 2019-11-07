/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

SHARED_RANDOM_INTERFACE(shogun::Machine)
%shared_ptr(shogun::Machine)
%shared_ptr(shogun::LinearMachine)
%shared_ptr(shogun::DistanceMachine)
%shared_ptr(shogun::IterativeMachine<LinearMachine>)

