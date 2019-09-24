/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */

%{
    #include <shogun/lib/observers/ParameterObserver.h>
    #include <shogun/lib/observers/ObservedValue.h>
%}

%shared_ptr(shogun::ParameterObserver)
%shared_ptr(shogun::ParameterObserverInterface)
%shared_ptr(shogun::ParameterObserverCV)
%shared_ptr(shogun::ObservedValue)

%include <shogun/lib/observers/ParameterObserver.h>
%include <shogun/lib/observers/ObservedValue.h>
