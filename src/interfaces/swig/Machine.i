/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

/*%warnfilter(302) apply;
%warnfilter(302) apply_generic;*/

%shared_ptr(shogun::Seedable<shogun::Machine>)
%shared_ptr(shogun::RandomMixin<shogun::Machine, std::mt19937_64>)
%shared_ptr(shogun::Machine)
%shared_ptr(shogun::LinearMachine)
%shared_ptr(shogun::DistanceMachine)
%shared_ptr(shogun::IterativeMachine<LinearMachine>)

