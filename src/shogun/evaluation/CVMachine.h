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
#include <thread>
#include <shogun/evaluation/CrossValidation.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
    /** @brief CVMachine is a machine that find the best parameters 
     * from the given parameter list
     */
    template<typename MachineType>
    class CVMachine : public SGObject
    {   
    public:
        CVMachine(std::vector<std::pair<std::string_view, std::vector<float64_t>>> params, 
            std::shared_ptr<CrossValidation> cross_validater): 
            m_params(std::move(params)), m_cross_validater(cross_validater)
        {
        }

        void fit(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
        {
            compute_params_pair();
            m_machine = m_cross_validater->get_machine();
            std::vector<std::pair<float64_t, 
                std::vector<std::pair<std::string_view, 
                    float64_t>>>> best_param_candidates(m_params_pair.size());
            
            const int32_t num_threads = env()->get_num_threads();

            if(num_threads > 1) 
            {
                std::vector<std::thread> threads;
                threads.reserve(m_params_pair.size());
                int32_t machine_per_thread = m_params_pair.size() / num_threads;
                if(machine_per_thread < 1) {
                    machine_per_thread = 1;
                }
                for(int t =0; t< std::min(num_threads, static_cast<int32_t>(m_params_pair.size())); t++) {
                    threads.emplace_back(
                        [&](int32_t start, int32_t end) 
                        {
                            for(int32_t i = start; i < end; i++) 
                            {
                                auto param_pair = m_params_pair[i];
                                auto result = cross_validate_one_machine(param_pair, data, labs);
                                best_param_candidates[i] = std::make_pair(result, param_pair);
                            }
                        }, t, t + machine_per_thread);
                }
                for(auto&& thread: threads) {
                    thread.join();
                }
                for(int i = machine_per_thread * num_threads; i < m_params_pair.size(); i++) 
                {
                    auto param_pair = m_params_pair[i];
                    auto result = cross_validate_one_machine(param_pair, data, labs);
                    best_param_candidates[i] = std::make_pair(result, param_pair);
                }
            }
            else 
            {
                for (int i = 0; i< m_params_pair.size(); i++) 
                {
                    auto param_pair = m_params_pair[i];
                    auto result = cross_validate_one_machine(param_pair, data, labs);
                    best_param_candidates[i] = std::make_pair(result, param_pair);
                }
            }
            std::sort(best_param_candidates.begin(), best_param_candidates.end(),
                [](auto&& lhs, auto && rhs)
                {
                    return lhs.first < rhs.first;
                });
            const auto best_param_pair = best_param_candidates[0];
            m_best_param = best_param_pair.second;

            //train m_machine with best parameter
            for(auto&& [param_name, param_value]: m_best_param)
            {
                m_machine->put(param_name, param_value);
            }
            m_machine->set_labels(labs);
            m_machine->train(data);
        }

        const char* get_name() const override
        {
            return "CVMachine";
        }

        std::shared_ptr<Labels> apply(std::shared_ptr<Features> data)
        {
            return m_machine->apply(data);
        }

        std::vector<std::pair<std::string_view, float64_t>> get_best_parameter() const
        {
            return m_best_param;
        }
    private:
        
        float64_t cross_validate_one_machine(const std::vector<
                        std::pair<std::string_view, float64_t>>& param_pair,
                        const std::shared_ptr<Features>& data, const std::shared_ptr<Labels> &labs) 
        {
            auto machine = make_clone(m_machine);
            auto cross_validater = make_clone(m_cross_validater);
            cross_validater->put("machine", machine);

            for(auto&& [param_name, param_value]: std::as_const(param_pair))
            {
                machine->put(param_name, param_value);
            }
            return cross_validater->evaluate(data, labs)
                                 -> template as<CrossValidationResult>()->get_mean();
        }
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
        //converted parameters, such as {{{"C1", 1}, {"C2", 1}}, {{"C1", 1}, {"C2", 2}},
        // {{"C1", 2}, {"C2", 1}}, {{"C1", 2}, {"C2", 2}}}
        std::vector<std::vector<std::pair<std::string_view, float64_t>>> m_params_pair;
        //given parameters, such as {{"C1", {1,2}}, {"C2"}, {1, 2}}
        std::vector<std::pair<std::string_view, std::vector<float64_t>>> m_params;
        std::shared_ptr<CrossValidation> m_cross_validater;
        std::shared_ptr<Machine> m_machine;
        std::vector<std::pair<std::string_view, float64_t>> m_best_param;
    };
}
#endif