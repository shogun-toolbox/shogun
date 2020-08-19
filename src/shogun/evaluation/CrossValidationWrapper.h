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
#include <utility>
#include <shogun/evaluation/CrossValidation.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
    template<typename MachineType>
    class CrossValidationWrapper : public SGObject
    {   
    public:
        CrossValidationWrapper(std::vector<std::pair<std::string_view, std::vector<float64_t>>> params, 
            std::shared_ptr<CrossValidation> cross_validater): 
            m_params(std::move(params)), m_cross_validater(cross_validater)
        {
        }

        void fit(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
        {
            compute_params_pair();
            m_machine = m_cross_validater->get_machine();
            std::priority_queue<std::pair<float64_t, 
                std::vector<std::pair<std::string_view, float64_t>>>> best_param_candidates;
            for(auto&& param_pair: std::as_const(m_params_pair))
            {
                for(auto&& [param_name, param_value]: std::as_const(param_pair))
                {
                    m_machine->put(param_name, param_value);
                }
                auto result = m_cross_validater->evaluate(data, labs)
                            ->as<CrossValidationResult>()->get_mean();
                best_param_candidates.push(std::make_pair(result, param_pair));
            }
            const auto best_param_pair = best_param_candidates.top();
            m_best_param = best_param_pair.second;
            for(auto&& [param_name, param_value]: std::as_const(m_best_param))
            {
                m_machine->put(param_name, param_value);
            }
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

        std::vector<std::pair<std::string_view, float64_t>> get_best_parameter() const
        {
            return m_best_param;
        }
    private:

         void compute_params_pair()
        {
            compute_params_pair_impl(0, m_params.size(), {});
        }
        void compute_params_pair_impl(int idx, int size, 
            std::vector<std::pair<std::string_view, float64_t>> m_param_pair)
        {
            if(idx == size) {
                m_params_pair.push_back(m_param_pair);
                return ;
            }
            const auto& param_name = m_params[idx].first;
            const auto& elem = m_params[idx].second;
            for(int i = 0; i<elem.size(); i++)
            {
                m_param_pair.push_back(std::make_pair(param_name,elem[i]));
                compute_params_pair_impl(idx + 1, size, m_param_pair);
                m_param_pair.pop_back();
            }
        }

        std::vector<std::vector<std::pair<std::string_view, float64_t>>> m_params_pair; 
        std::vector<std::pair<std::string_view, std::vector<float64_t>>> m_params;
        const std::shared_ptr<CrossValidation> m_cross_validater;
        std::shared_ptr<Machine> m_machine;
        std::vector<std::pair<std::string_view, float64_t>> m_best_param;
    };
}
#endif