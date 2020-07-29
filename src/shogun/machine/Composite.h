/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#ifndef __COMPOSITE_H
#define __COMPOSITE_H
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>
#include <shogun/transformer/Transformer.h>
#include <shogun/ensemble/CombinationRule.h>
#include <shogun/machine/EnsembleMachine.h>
#include <vector>
#include <memory>

namespace shogun{
class Composite : public Machine {
public:
    Composite() = default;

    ~Composite() = default;
    std::shared_ptr<Composite> with(const std::shared_ptr<Machine>& machine){
        m_ensemble_machine->add_machine(machine);
        return shared_from_this()->as<Composite>(); 
    }

    std::shared_ptr<Composite> then(const std::shared_ptr<CombinationRule>& rule){
        m_ensemble_machine->set_combination_rule(rule);
        return shared_from_this()->as<Composite>();
    }

    std::shared_ptr<EnsembleMachine> train(const std::shared_ptr<Features>& data, 
        const std::shared_ptr<Labels>& labs)
    {
        m_ensemble_machine->train(data, labs);
        return m_ensemble_machine;
    }
    
private:
    std::shared_ptr<EnsembleMachine> m_ensemble_machine 
            = std::make_shared<EnsembleMachine>();
};
} //nampsapce 
#endif
