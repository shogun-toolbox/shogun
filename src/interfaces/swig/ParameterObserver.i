/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */

%{
    #include <shogun/lib/parameter_observers/ParameterObserver.h>
%}

%include <shogun/lib/parameter_observers/ParameterObserver.h>

namespace shogun {

%template(get_real) ParameterObserver::get_observation<float64_t>;
%template(get_int) ParameterObserver::get_observation<int32_t>;
%template(get_real_matrix) ParameterObserver::get_observation<SGMatrix<float64_t>>;
%template(get_char_string_list) ParameterObserver::get_observation<SGStringList<char>>;
%template(get_word_string_list) ParameterObserver::get_observation<SGStringList<uint16_t>>;
%template(get_option) ParameterObserver::get_observation<std::string>;
#ifndef SWIGJAVA
%template(get_real_vector) ParameterObserver::get_observation<SGVector<float64_t>>;
%template(get_int_vector) ParameterObserver::get_observation<SGVector<int32_t>>;
#else // SWIGJAVA
%template(get_real_vector) ParameterObserver::get_vector_as_matrix_dispatcher<SGMatrix<float64_t>>;
%template(get_int_vector) ParameterObserver::get_vector_as_matrix_dispatcher<SGMatrix<int32_t>>;
#endif // SWIGJAVA

%template(get_CV_storage) ParameterObserver::get_observation<CrossValidationStorage>;
%template(CrossValidationObservations) vector<CrossValidationStorage>;



}
