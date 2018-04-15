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

class CDynamicObjectArray;
class CLabels;

/** Base class of all Multiclass Machines.
 */
class CBaseMulticlassMachine: public CMachine
{
public:
    /** constructor */
	CBaseMulticlassMachine();

    /** destructor */
	virtual ~CBaseMulticlassMachine();

    /** get name */
    virtual const char* get_name() const { return "BaseMulticlassMachine"; }

	/** get number of machines
	 *
	 * @return number of machines
	 */
	int32_t get_num_machines() const;

	/** get problem type */
	virtual EProblemType get_machine_problem_type() const;

	/** check whether the labels is valid.
	 *
	 * @param lab the labels being checked, guaranteed to be non-NULL
	 */
	virtual bool is_label_valid(CLabels *lab) const;

protected:

	/** machines */
	CDynamicObjectArray *m_machines;
};

} /* shogun */

#endif /* end of include guard: BASEMULTICLASSMACHINE_H__ */

