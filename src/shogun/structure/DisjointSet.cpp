/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Bjoern Esser
 */

#include <shogun/structure/DisjointSet.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

DisjointSet::DisjointSet()
	: SGObject()
{
	SG_UNSTABLE("CDisjointSet::CDisjointSet()", "\n");

	init();
}

DisjointSet::DisjointSet(int32_t num_elements)
	: SGObject()
{
	init();
	m_num_elements = num_elements;
	m_parent = SGVector<int32_t>(num_elements);
	m_rank = SGVector<int32_t>(num_elements);
}

void DisjointSet::init()
{
	SG_ADD(&m_num_elements, "num_elements", "Number of elements");
	SG_ADD(&m_parent, "parent", "Parent pointers");
	SG_ADD(&m_rank, "rank", "Rank of each element");
	SG_ADD(&m_is_connected, "is_connected", "Whether disjoint sets have been linked");

	m_is_connected = false;
	m_num_elements = -1;
}

void DisjointSet::make_sets()
{
	REQUIRE(m_num_elements > 0, "%s::make_sets(): m_num_elements <= 0.\n", get_name());

	m_parent.range_fill();
	m_rank.zero();
}

int32_t DisjointSet::find_set(int32_t x)
{
	ASSERT(x >= 0 && x < m_num_elements);

	// path compression
	if (x != m_parent[x])
		m_parent[x] = find_set(m_parent[x]);

	return m_parent[x];
}

int32_t DisjointSet::link_set(int32_t xroot, int32_t yroot)
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

bool DisjointSet::union_set(int32_t x, int32_t y)
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

bool DisjointSet::is_same_set(int32_t x, int32_t y)
{
	ASSERT(x >= 0 && x < m_num_elements);
	ASSERT(y >= 0 && y < m_num_elements);

	if (find_set(x) == find_set(y))
		return true;

	return false;
}

int32_t DisjointSet::get_unique_labeling(SGVector<int32_t> out_labels)
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

int32_t DisjointSet::get_num_sets()
{
	REQUIRE(m_num_elements > 0, "%s::get_num_sets(): m_num_elements <= 0.\n", get_name());

	return get_unique_labeling(SGVector<int32_t>(m_num_elements));
}

bool DisjointSet::get_connected()
{
	return m_is_connected;
}

void DisjointSet::set_connected(bool is_connected)
{
	m_is_connected = is_connected;
}

