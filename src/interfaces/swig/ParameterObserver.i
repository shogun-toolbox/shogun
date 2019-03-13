/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */

%{
    #include <shogun/lib/parameter_observers/ObservedValue.h>
    #include <shogun/lib/parameter_observers/ParameterObserver.h>
%}

%include <shogun/lib/parameter_observers/ObservedValue.h>
%include <shogun/lib/parameter_observers/ParameterObserver.h>

namespace shogun {

%template(SomeObservedValue) Some<ObservedValue>;
%template(get_CV_storage) CSGObject::get<CrossValidationStorage*, void>;

}
