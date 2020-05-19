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
#include <shogun/machine/BaggingMachine.h>
#include <vector>
#include <memory>

namespace shogun{
class Composite : public Machine {
public:
    Composite() = default;

    Composite(std::vector<std::shared_ptr<Transformer>>&& trans ) :m_trans(std::move(trans)){}

    ~Composite() = default;
    std::shared_ptr<Composite> with(std::shared_ptr<Machine>);
    std::shared_ptr<Composite> then(std::shared_ptr<CombinationRule>);
    std::shared_ptr<BaggingMachine> fit(std::shared_ptr<Features>, std::shared_ptr<Labels>);
private:
    /** Transformer to transform data*/
    std::vector<std::shared_ptr<Transformer>> m_trans;
    /** machine to be trained in BaggingMachine*/
    std::vector<std::shared_ptr<Machine>> m_machines;
    
    std::shared_ptr<CombinationRule> m_rule;
};
} //nampsapce 
#endif
