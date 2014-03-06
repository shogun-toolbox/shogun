/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef GENERICTREEMACHINE_H__
#define GENERICTREEMACHINE_H__

#include <shogun/machine/BaseMulticlassMachine.h>
#include <shogun/multiclass/tree/GenericTreeMachineNode.h>

namespace shogun
{

/** @brief class GenericTreeMachine, a base class
 * for tree based multiclass classifiers where more
 * than 2 children may be required
 */
template <class T> class CGenericTreeMachine : public CBaseMulticlassMachine
{

public:
	/** constructor */
	CGenericTreeMachine():CBaseMulticlassMachine()
	{
		m_root=NULL;
		SG_ADD((CSGObject**)&m_root,"m_root", "tree structure", MS_NOT_AVAILABLE);
	}

	/** destructor */
	virtual ~CGenericTreeMachine()
	{
		SG_UNREF(m_root);
	}

	/** get name */
	virtual const char* get_name() const { return "GenericTreeMachine"; }

	/** set root
	 * @param root the root node of the tree
	 */
	void set_root(CGenericTreeMachineNode<T>* root)
	{
		m_root=root;
	}

	/** get root
	 * @return root the root node of the tree
	 */
	CGenericTreeMachineNode<T>* get_root()
	{
		return m_root;
	}

protected:
	/** tree root */
	CGenericTreeMachineNode<T>* m_root;
};

} /* shogun */

#endif /* GENERICTREEMACHINE_H__ */

