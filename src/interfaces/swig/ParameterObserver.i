/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */
%shared_ptr(shogun::ParameterObserverInterface)
%shared_ptr(shogun::ParameterObserverCV)

%{
    #include <shogun/lib/observers/ParameterObserver.h>
    #include <shogun/lib/observers/ObservedValue.h>
%}

%include <shogun/lib/observers/ParameterObserver.h>
%include <shogun/lib/observers/ObservedValue.h>
