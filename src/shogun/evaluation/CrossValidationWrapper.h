/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */

#ifndef CROSSVALIDATATION_WRAPPER_H__
#define CROSSVALIDATATION_WRAPPER_H__

#include <vector>
#include <shogun/machine/Machine.h>
#include <type_traits>
#include <string_view>
#include <memory>
#include <queue>
#include <shogun/evaluation/CrossValidation.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
    template<typename MachineType>
    class CrossValidationWrapper : public SGObject
    {   
    public:
        CrossValidationWrapper(std::pair<std::string_view, std::vector<float64_t>> params, 
            std::shared_ptr<CrossValidation> cross_validater): 
            m_params(params), m_cross_validater(cross_validater)
        {
        }

        void fit(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
        {
            const auto& param_name = m_params.first;
            m_machine = m_cross_validater->get_machine();
            std::priority_queue<std::pair<float64_t, float64_t>> best_param_candidates;
            for(const float64_t& param : m_params.second)
            {
                m_machine->put(param_name, param);
                auto result = m_cross_validater->evaluate(data, labs)
                            ->as<CrossValidationResult>()->get_mean();
                best_param_candidates.push(std::make_pair(result, param));
            }
            const auto best_param_pair = best_param_candidates.top();
            m_best_param = best_param_pair.second;
            m_machine ->put(param_name, m_best_param);
            m_machine->set_labels(labs);
            m_machine->train(data);
        }

        const char* get_name() const override
        {
            return "CrossValidationWrapper";
        }
        std::shared_ptr<Labels> apply(std::shared_ptr<Features> data)
        {
            return m_machine ->apply(data);
        }
        float64_t get_best_parameter() const
        {
            return m_best_param;
        }
    private:
        std::pair<std::string_view, std::vector<float64_t>> m_params;
        const std::shared_ptr<CrossValidation> m_cross_validater;
        std::shared_ptr<Machine> m_machine;
        float64_t m_best_param;
    };
}
#endif