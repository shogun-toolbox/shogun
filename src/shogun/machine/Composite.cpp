/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include <memory>

#include <shogun/machine/Composite.h>

using namespace shogun;


 std::shared_ptr<Composite> Composite::with(std::shared_ptr<Machine> machine){
            m_machines.push_back(machine);
            return shared_from_this()->as<Composite>();
    }
std::shared_ptr<Composite> Composite::then(std::shared_ptr<CombinationRule> rule){
            m_rule = rule;
            return shared_from_this()->as<Composite>();
}
std::shared_ptr<BaggingMachine> Composite::fit(std::shared_ptr<Features> data, std::shared_ptr<Labels> y){
    auto curr_data = data;
    for(auto && tran: m_trans){
        curr_data = tran->transform(curr_data);
    }
    auto bagging = std::make_shared<BaggingMachine>(); 
    for(auto&& machine : m_machines){
        bagging->set_labels(y);
        bagging->set_machine(machine);
        bagging->train(data);
    }
    return bagging;
}

