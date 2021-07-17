/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Thoralf Klein, Bjoern Esser, 
 *          Sergey Lisitsyn, Soeren Sonnenburg
 */

#ifndef BASEMULTICLASSMACHINE_H__
#define BASEMULTICLASSMACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/machine/Machine.h>

namespace shogun
{

class Labels;

/** Base class of all Multiclass Machines.
 */
class BaseMulticlassMachine: public Machine
{
public:
    /** constructor */
	BaseMulticlassMachine();

    /** destructor */
	~BaseMulticlassMachine() override;

    /** get name */
    const char* get_name() const override { return "BaseMulticlassMachine"; }

	/** get number of machines
	 *
	 * @return number of machines
	 */
	int32_t get_num_machines() const;

	/** get problem type */
	EProblemType get_machine_problem_type() const override;

	/** check whether the labels is valid.
	 *
	 * @param lab the labels being checked, guaranteed to be non-NULL
	 */
	bool is_label_valid(std::shared_ptr<Labels> lab) const override;

protected:

	/** machines */
	std::vector<std::shared_ptr<Machine>> m_machines;

#ifndef SWIG
public:
     static constexpr std::string_view kMachines = "machines";

} /* shogun */

#endif /* end of include guard: BASEMULTICLASSMACHINE_H__ */

