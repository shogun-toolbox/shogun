/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2014 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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

#include <shogun/features/SubsetStack.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CSubsetStack::CSubsetStack()
{
	init();
}

CSubsetStack::CSubsetStack(const CSubsetStack& other)
{
	init();

	for (int32_t i=0; i < other.m_active_subsets_stack->get_num_elements(); ++i)
	{
		m_active_subset=(CSubset*)other.m_active_subsets_stack->get_element(i);
		m_active_subsets_stack->append_element(m_active_subset);
	}
}

CSubsetStack::~CSubsetStack()
{
	SG_UNREF(m_active_subsets_stack);
	SG_UNREF(m_active_subset);
}

void CSubsetStack::remove_all_subsets()
{
	/* delete all active subsets, backwards due to DynArray implementation */
	for (index_t i=m_active_subsets_stack->get_num_elements()-1; i>=0; --i)
		m_active_subsets_stack->delete_element(i);

	SG_UNREF(m_active_subset);
}

void CSubsetStack::init()
{
	SG_ADD((CSGObject**)&m_active_subset, "active_subset",
			"Currently active subset", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_active_subsets_stack, "active_subsets_stack",
			"Stack of active subsets", MS_NOT_AVAILABLE);

	m_active_subset=NULL;
	m_active_subsets_stack=new CDynamicObjectArray();
	SG_REF(m_active_subsets_stack);
}

void CSubsetStack::add_subset(SGVector<index_t> subset)
{
	/* if there are already subsets on stack, do some legality checks */
	if (m_active_subsets_stack->get_num_elements())
	{
		/* check that subsets may only be smaller or equal than existing */
		CSubset* latest=(CSubset*)m_active_subsets_stack->get_last_element();
		if (subset.vlen>latest->m_subset_idx.vlen)
		{
			subset.display_vector("subset");
			latest->m_subset_idx.display_vector("last on stack");
			SG_ERROR("%s::add_subset(): Provided index vector is "
					"larger than the subsets on the stubset stack!\n", get_name());
		}

		/* check for range of indices */
		index_t max_index=CMath::max<index_t>(subset.vector, subset.vlen);
		if (max_index>=latest->m_subset_idx.vlen)
		{
			subset.display_vector("subset");
			latest->m_subset_idx.display_vector("last on stack");
			SG_ERROR("%s::add_subset(): Provided index vector contains"
					" indices larger than possible range!\n", get_name());
		}

		/* clean up */
		SG_UNREF(latest);
	}

	/* active subset will be changed anyway, no setting to NULL */
	SG_UNREF(m_active_subset);

	/* two cases: stack is empty/stack is not empty */
	if (m_active_subsets_stack->get_num_elements())
	{
		/* if there are already subsets, we need to map given one through
		 * existing ones */

		/* get latest current subset */
		CSubset* latest=(CSubset*)m_active_subsets_stack->get_last_element();

		/* create new index vector */
		SGVector<index_t> new_active_subset=SGVector<index_t>(subset.vlen);

		/* using the latest current subset, transform all indices by the latest
		 * added subset (dynamic programming greets you) */
		for (index_t i=0; i<subset.vlen; ++i)
		{
			new_active_subset.vector[i]=
					latest->m_subset_idx.vector[subset.vector[i]];
		}

		/* replace active subset */
		m_active_subset=new CSubset(new_active_subset);
		SG_REF(m_active_subset);
		SG_UNREF(latest);
	}
	else
	{
		/* just use plain given subset since there is nothing to map */
		m_active_subset=new CSubset(subset);
		SG_REF(m_active_subset);
	}

	/* add current active subset on stack of active subsets in any case */
	m_active_subsets_stack->append_element(m_active_subset);
}

void CSubsetStack::add_subset_in_place(SGVector<index_t> subset)
{
	SG_NOTIMPLEMENTED;
}

void CSubsetStack::remove_subset()
{
	index_t num_subsets=m_active_subsets_stack->get_num_elements();
	if (num_subsets)
	{
		/* unref current subset */
		SG_UNREF(m_active_subset);
		m_active_subset=NULL;

		/* delete last element on stack */
		if (num_subsets>=1)
		{
			index_t last_idx=m_active_subsets_stack->get_num_elements()-1;
			m_active_subsets_stack->delete_element(last_idx);
		}

		/* if there are subsets left on stack, set the next one as active */
		if (num_subsets>1)
		{
			/* use new last element on stack as active subset */
			index_t last_idx=m_active_subsets_stack->get_num_elements()-1;
			m_active_subset=(CSubset*)
					m_active_subsets_stack->get_element(last_idx);
		}

		/* otherwise, active subset is just empty */
	}
	
	/* do nothing if nothing on stack */
}
