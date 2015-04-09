/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <shogun/structure/DisjointSet.h>
#include <shogun/base/Parameter.h>

#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

CDisjointSet::CDisjointSet()
	: CSGObject()
{
	SG_UNSTABLE("CDisjointSet::CDisjointSet()", "\n");

	init();
}

CDisjointSet::CDisjointSet(int32_t num_elements)
	: CSGObject()
{
	init();
	m_num_elements = num_elements;
	m_parent = SGVector<int32_t>(num_elements);
	m_rank = SGVector<int32_t>(num_elements);
}

void CDisjointSet::init()
{
	SG_ADD(&m_num_elements, "num_elements", "Number of elements", MS_NOT_AVAILABLE);
	SG_ADD(&m_parent, "parent", "Parent pointers", MS_NOT_AVAILABLE);
	SG_ADD(&m_rank, "rank", "Rank of each element", MS_NOT_AVAILABLE);
	SG_ADD(&m_is_connected, "is_connected", "Whether disjoint sets have been linked", MS_NOT_AVAILABLE);

	m_is_connected = false;
	m_num_elements = -1;
}

void CDisjointSet::make_sets()
{
	REQUIRE(m_num_elements > 0, "%s::make_sets(): m_num_elements <= 0.\n", get_name());

	linalg::range_fill<linalg::Backend::NATIVE>(m_parent);
	m_rank.zero();
}

int32_t CDisjointSet::find_set(int32_t x)
{
	ASSERT(x >= 0 && x < m_num_elements);

	// path compression
	if (x != m_parent[x])
		m_parent[x] = find_set(m_parent[x]);

	return m_parent[x];
}

int32_t CDisjointSet::link_set(int32_t xroot, int32_t yroot)
{
	ASSERT(xroot >= 0 && xroot < m_num_elements);
	ASSERT(yroot >= 0 && yroot < m_num_elements);
	ASSERT(m_parent[xroot] == xroot && m_parent[yroot] == yroot);
	ASSERT(xroot != yroot);

	// union by rank
	if (m_rank[xroot] > m_rank[yroot])
	{
		m_parent[yroot] = xroot;
		return xroot;
	}
	else
	{
		m_parent[xroot] = yroot;
		if (m_rank[xroot] == m_rank[yroot])
			m_rank[yroot] += 1;

		return yroot;
	}
}

bool CDisjointSet::union_set(int32_t x, int32_t y)
{
	ASSERT(x >= 0 && x < m_num_elements);
	ASSERT(y >= 0 && y < m_num_elements);

	int32_t xroot = find_set(x);
	int32_t yroot = find_set(y);

	if (xroot == yroot)
		return true;

	link_set(xroot, yroot);
	return false;
}

bool CDisjointSet::is_same_set(int32_t x, int32_t y)
{
	ASSERT(x >= 0 && x < m_num_elements);
	ASSERT(y >= 0 && y < m_num_elements);

	if (find_set(x) == find_set(y))
		return true;

	return false;
}

int32_t CDisjointSet::get_unique_labeling(SGVector<int32_t> out_labels)
{
	REQUIRE(m_num_elements > 0, "%s::get_unique_labeling(): m_num_elements <= 0.\n", get_name());

	if (out_labels.size() != m_num_elements)
		out_labels.resize_vector(m_num_elements);

	SGVector<int32_t> roots(m_num_elements);
	SGVector<int32_t> flags(m_num_elements);
	SGVector<int32_t>::fill_vector(flags.vector, flags.vlen, -1);
	int32_t unilabel = 0;

	for (int32_t i = 0; i < m_num_elements; i++)
	{
		roots[i] = find_set(i);
		// if roots[i] never be found
		if (flags[roots[i]] < 0)
			flags[roots[i]] = unilabel++;
	}

	for (int32_t i = 0; i < m_num_elements; i++)
		out_labels[i] = flags[roots[i]];

	return unilabel;
}

int32_t CDisjointSet::get_num_sets()
{
	REQUIRE(m_num_elements > 0, "%s::get_num_sets(): m_num_elements <= 0.\n", get_name());

	return get_unique_labeling(SGVector<int32_t>(m_num_elements));
}

bool CDisjointSet::get_connected()
{
	return m_is_connected;
}

void CDisjointSet::set_connected(bool is_connected)
{
	m_is_connected = is_connected;
}

