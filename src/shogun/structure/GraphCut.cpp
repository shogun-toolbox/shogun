/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jiaolong Xu, Fernando Iglesias, Bjoern Esser
 */

#include <shogun/structure/GraphCut.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

GraphCut::GraphCut()
	: MAPInferImpl()
{
	unstable(SOURCE_LOCATION);

	init();
}

GraphCut::GraphCut(std::shared_ptr<FactorGraph> fg)
	: MAPInferImpl(fg)
{
	ASSERT(m_fg != NULL);

	init();
}

GraphCut::GraphCut(int32_t num_nodes, int32_t num_edges)
	: MAPInferImpl()
{
	init();

	m_num_nodes = num_nodes;
	// build s-t graph
	build_st_graph(m_num_nodes, num_edges);
}

GraphCut::~GraphCut()
{
	if (m_nodes!=NULL)
		SG_FREE(m_nodes);

	if (m_edges!=NULL)
		SG_FREE(m_edges);
}

void GraphCut::init()
{
	m_nodes = NULL;
	m_edges = NULL;
	m_edges_last = NULL;
	m_num_nodes = 0;
	m_num_edges = 0;

	m_active_first[0] = NULL;
	m_active_last[0] = NULL;
	m_active_first[1] = NULL;
	m_active_last[1] = NULL;
	m_orphan_first = NULL;
	m_orphan_last = NULL;

	m_timestamp = 0;
	m_flow = 0;
	m_map_energy = 0;

	if (m_fg == NULL)
		return;

	auto facs = m_fg->get_factors();

	SGVector<int32_t> cards = m_fg->get_cardinalities();

	for (int32_t i = 0; i < cards.size(); i++)
	{
		if (cards[i] != 2)
		{
			error("This implementation of the graph cut optimizer supports only binary variables.");
		}
	}

	m_num_factors_at_order = SGVector<int32_t> (4);
	m_num_factors_at_order.zero();

	for (int32_t i = 0; i < facs->get_num_elements(); i++)
	{
		auto fac = facs->get_element<Factor>(i);

		int32_t num_vars = fac->get_num_vars();



		if (num_vars > 3)
		{
			error("This implementation of the graph cut optimizer supports only factors of order <= 3.");
		}

		++m_num_factors_at_order[num_vars];

	}

	m_num_variables = m_fg->get_num_vars();
	int32_t max_num_edges = m_num_factors_at_order[2] + 3 * m_num_factors_at_order[3];
	m_num_nodes = m_num_variables + m_num_factors_at_order[3];

	// build s-t graph
	build_st_graph(m_num_nodes, max_num_edges);

	for (int32_t j = 0; j < m_fg->get_num_factors(); j++)
	{
		auto fac = facs->get_element<Factor>(j);
		add_factor(fac);

	}


}

void GraphCut::build_st_graph(int32_t num_nodes, int32_t num_edges)
{
	m_num_nodes = num_nodes;

	// allocate s-t graph
	m_nodes = SG_MALLOC(GCNode, m_num_nodes);
	m_edges = SG_MALLOC(GCEdge, 2 * num_edges);
	m_edges_last = m_edges;

	for (int32_t i = 0; i < m_num_nodes; i++)
	{
		m_nodes[i].id = i;
		m_nodes[i].tree_cap = 0;
		m_nodes[i].first = NULL;
	}

	m_num_edges = 0; // m_num_edges will be counted in add_edge()
	m_flow = 0;

	m_active_first[0] = NULL;
	m_active_last[0] = NULL;
	m_active_first[1] = NULL;
	m_active_last[1] = NULL;
	m_orphan_first = NULL;
	m_orphan_last = NULL;

	m_timestamp = 0;
}

void GraphCut::init_maxflow()
{
	GCNode* node_i;

	m_active_first[0] = NULL;
	m_active_last[0] = NULL;
	m_active_first[1] = NULL;
	m_active_last[1] = NULL;
	m_orphan_first = NULL;
	m_orphan_last = NULL;

	m_timestamp = 0;

	for (int32_t i = 0; i < m_num_nodes; i++)
	{
		node_i = m_nodes + i;
		node_i->next = NULL;
		node_i->timestamp = m_timestamp;

		if (node_i->tree_cap > 0)
		{
			// i is connected to the source
			node_i->type_tree = SOURCE;
			node_i->parent = TERMINAL_EDGE;
			set_active(node_i);
			node_i->dist_terminal = 1;
		}
		else if (node_i->tree_cap < 0)
		{
			// i is connected to the sink
			node_i->type_tree = SINK;
			node_i->parent = TERMINAL_EDGE;
			set_active(node_i);
			node_i->dist_terminal = 1;
		}
		else
		{
			node_i->parent = NULL;
		}
	}
}

float64_t GraphCut::inference(SGVector<int32_t> assignment)
{
	require(assignment.size() == m_fg->get_cardinalities().size(),
	        "{}::inference(): the output assignment should be prepared as"
	        "the same size as variables!", get_name());

	// compute max flow
	init_maxflow();
	compute_maxflow();

	for (int32_t vi = 0; vi < assignment.size(); vi++)
	{
		assignment[vi] = get_assignment(vi) == SOURCE ? 0 : 1;
	}

	m_map_energy = m_fg->evaluate_energy(assignment);
	SG_DEBUG("fg.evaluate_energy(assignment) = {}", m_fg->evaluate_energy(assignment));
	SG_DEBUG("minimized energy = {}", m_map_energy);

	return m_map_energy;
}

void GraphCut::add_factor(std::shared_ptr<Factor> factor)
{
	SGVector<int32_t> fcards = factor->get_cardinalities();

	for (int32_t i = 0; i < fcards.size(); i++)
	{
		ASSERT(fcards[i] == 2);
	}

	int32_t f_order = factor->get_num_vars();

	switch (f_order)
	{
	case 0:
		break;
	case 1:
	{
		SGVector<int32_t> fvars = factor->get_variables();
		SGVector<float64_t> fenrgs = factor->get_energies();
		ASSERT(fenrgs.size() == 2);
		int32_t var = fvars[0];
		float64_t v0 = fenrgs[0];
		float64_t v1 = fenrgs[1];

		if (v0 < v1)
		{
			add_tweights(var, v1 - v0, 0);
		}
		else
		{
			add_tweights(var, 0, v0 - v1);
		}
	}
	break;
	case 2:
	{
		SGVector<int32_t> fvars = factor->get_variables();
		SGVector<float64_t> fenrgs = factor->get_energies();
		int32_t var0 = fvars[0];
		int32_t var1 = fvars[1];
		float64_t A = fenrgs[0]; //E{0,0} = {y_var0, y_var1}
		float64_t B = fenrgs[2]; //E{0,1}
		float64_t C = fenrgs[1]; //E{1,0}
		float64_t D = fenrgs[3]; //E{1,1}

		// Added "truncation" code below to ensure regularity / submodularity
		if (A + D > C + B)
		{
			SG_DEBUG("Truncation is applied to ensure regularity / submodularity.");

			float64_t delta = A + D - C - B;
			float64_t subtrA = delta / 3;
			A = A - subtrA;
			C = C + subtrA;
			B = B + (delta - subtrA * 2) + 0.0001; // for numeric issue
		}

		// first variabe
		if (C > A)
		{
			add_tweights(var0, C - A, 0);
		}
		else
		{
			add_tweights(var0, 0, A - C);
		}
		// second varibale
		if (D > C)
		{
			add_tweights(var1, D - C, 0);
		}
		else
		{
			add_tweights(var1, 0, C - D);
		}

		// submodular term
		float64_t term = B + C - A - D;

		// term >= 0 is the regularity condition.
		// It is the sufficient and necessary condition for any function to be graph-representable
		if (term < 0)
		{
			error("\nRegularity condition is not satisfied");
		}

		add_edge(var0, var1, term, 0);
	}
	break;
	case 3:
	{
		SGVector<int32_t> fvars = factor->get_variables();
		SGVector<float64_t> fenrgs = factor->get_energies();
		int32_t var0 = fvars[0];
		int32_t var1 = fvars[1];
		int32_t var2 = fvars[2];
		float64_t A = fenrgs[0]; //{0,0,0}
		float64_t E = fenrgs[1]; //{1,0,0}
		float64_t C = fenrgs[2]; //{0,1,0}
		float64_t G = fenrgs[3]; //{1,1,0}
		float64_t B = fenrgs[4]; //{0,0,1}
		float64_t F = fenrgs[5]; //{1,0,1}
		float64_t D = fenrgs[6]; //{0,1,1}
		float64_t H = fenrgs[7]; //{1,1,1}

		int32_t id = get_tripleId(fvars);
		float64_t P = (A + D + F + G) - (B + C + E + H);

		if (P >= 0.0)
		{
			if (F - B >= 0)
			{
				add_tweights(var0, F - B, 0);
			}
			else
			{
				add_tweights(var0, 0, B - F);
			}

			if (G - E >= 0)
			{
				add_tweights(var1, G - E, 0);
			}
			else
			{
				add_tweights(var1, 0, E - G);
			}

			if (D - C >= 0)
			{
				add_tweights(var2, D - C, 0);
			}
			else
			{
				add_tweights(var2, 0, C - D);
			}

			add_edge(var1, var2, B + C - A - D, 0);
			add_edge(var2, var0, B + E - A - F, 0);
			add_edge(var0, var1, C + E - A - G, 0);

			add_edge(var0, id, P, 0);
			add_edge(var1, id, P, 0);
			add_edge(var2, id, P, 0);
			add_edge(id, 1, P, 0);
		}
		else
		{
			if (C - G >= 0)
			{
				add_tweights(var0, 0, C - G);
			}
			else
			{
				add_tweights(var0, G - C, 0);
			}

			if (B - D >= 0)
			{
				add_tweights(var1, 0, B - D);
			}
			else
			{
				add_tweights(var1, D - B, 0);
			}

			if (E - F >= 0)
			{
				add_tweights(var2, 0, E - F);
			}
			else
			{
				add_tweights(var2, F - E, 0);
			}

			add_edge(var2, var1, F + G - E - H, 0);
			add_edge(var0, var2, D + G - C - H, 0);
			add_edge(var1, var0, D + F - B - H, 0);

			add_edge(id, var0, -P, 0);
			add_edge(id, var1, -P, 0);
			add_edge(id, var2, -P, 0);
			add_tweights(id, -P, 0);
		}
	}
	break;
	default:
		error("This implementation of the graph cut optimizer does not support factors of order > 3.");
		break;
	}
}

int32_t GraphCut::get_tripleId(SGVector<int32_t> triple)
{
	// search for triple in list
	int32_t counter = m_num_variables;

	for (int32_t i = 0; i < m_triple_list.get_num_elements(); i++)
	{
		SGVector<int32_t> vec = m_triple_list[i];

		if (triple[0] == vec[0] && triple[1] == vec[1] && triple[2] == vec[2])
		{
			return counter;
		}

		m_num_variables++;
	}
	// add triple to list
	m_triple_list.push_back(triple);

	ASSERT(counter - m_num_variables < m_num_factors_at_order[3]);

	return counter;
}

void GraphCut::add_tweights(int32_t i, float64_t cap_source, float64_t cap_sink)
{
	ASSERT(i >= 0 && i < m_num_nodes);

	float64_t delta = m_nodes[i].tree_cap;

	if (delta > 0)
	{
		cap_source += delta;
	}
	else
	{
		cap_sink -= delta;
	}

	m_flow += (cap_source < cap_sink) ? cap_source : cap_sink;

	m_nodes[i].tree_cap = cap_source - cap_sink;
}

void GraphCut::add_edge(int32_t i, int32_t j, float64_t capacity, float64_t reverse_capacity)
{
	ASSERT(i >= 0 && i < m_num_nodes);
	ASSERT(j >= 0 && j < m_num_nodes);
	ASSERT(i != j);
	ASSERT(capacity >= 0);
	ASSERT(reverse_capacity >= 0);

	GCEdge* e = m_edges_last++;
	e->id = m_num_edges++;
	GCEdge* e_rev = m_edges_last++;
	e_rev->id = m_num_edges++;

	GCNode* node_i = m_nodes + i;
	GCNode* node_j = m_nodes + j;

	e->reverse = e_rev;
	e_rev->reverse = e;
	e->next = node_i->first;
	node_i->first = e;
	e_rev->next = node_j->first;
	node_j->first = e_rev;
	e->head = node_j;
	e_rev->head = node_i;
	e->residual_capacity = capacity;
	e_rev->residual_capacity = reverse_capacity;
}

void GraphCut::set_active(GCNode* node_i)
{
	if (node_i->next == NULL)
	{
		// it's not in the list yet
		if (m_active_last[1])
		{
			m_active_last[1]->next = node_i;
		}
		else
		{
			m_active_first[1] = node_i;
		}

		m_active_last[1] = node_i;
		node_i->next = node_i;
	}
}

GCNode* GraphCut::next_active()
{
	// Returns the next active node. If it is connected to the sink,
	// it stays in the list, otherwise it is removed from the list.
	GCNode* node_i;

	while (true)
	{
		if ((node_i = m_active_first[0]) == NULL)
		{
			m_active_first[0] = node_i = m_active_first[1];
			m_active_last[0]  = m_active_last[1];
			m_active_first[1] = NULL;
			m_active_last[1]  = NULL;

			if (node_i == NULL)
			{
				return NULL;
			}
		}

		// remove it from the active list
		if (node_i->next == node_i)
		{
			m_active_first[0] = NULL;
			m_active_last[0] = NULL;
		}
		else
		{
			m_active_first[0] = node_i->next;
		}

		node_i->next = NULL;

		// a node in the list is active iff it has a parent
		if (node_i->parent != NULL)
		{
			return node_i;
		}
	}
}

float64_t GraphCut::compute_maxflow()
{
	GCNode* current_node = NULL;
	bool active_set_found = true;

	// start the main loop
	while (true)
	{
		if (env()->io()->get_loglevel() <= io::MSG_DEBUG)
			test_consistency(current_node);

		GCEdge* connecting_edge;

		// find a path from source to sink
		active_set_found = grow(connecting_edge, current_node);

		if (!active_set_found)
		{
			break;
		}

		if (connecting_edge == NULL)
		{
			continue;
		}

		m_timestamp++;

		// augment that path
		augment_path(connecting_edge);

		// adopt orphans, rebuild the search tree structure
		adopt();
	}

	if (env()->io()->get_loglevel() <= io::MSG_DEBUG)
		test_consistency();

	return m_flow;
}

bool GraphCut::grow(GCEdge* &edge, GCNode* &current_node)
{
	GCNode* node_i, *node_j;

	if ((node_i = current_node) != NULL)
	{
		node_i->next = NULL; // remove active flag

		if (node_i->parent == NULL)
		{
			node_i = NULL;
		}
	}

	if (node_i == NULL && (node_i = next_active()) == NULL)
	{
		return false;
	}

	if (node_i->type_tree == SOURCE)
	{
		// grow source tree
		for (edge = node_i->first; edge != NULL; edge = edge->next)
		{
			if (edge->residual_capacity)
			{
				node_j = edge->head;

				if (node_j->parent == NULL)
				{
					node_j->type_tree = SOURCE;
					node_j->parent = edge->reverse;
					node_j->timestamp = node_i->timestamp;
					node_j->dist_terminal = node_i->dist_terminal + 1;
					set_active(node_j);
				}
				else if (node_j->type_tree == SINK)
				{
					break;
				}
				else if (node_j->timestamp <= node_i->timestamp && node_j->dist_terminal > node_i->dist_terminal)
				{
					// heuristic - trying to make the distance from j to the source shorter
					node_j->parent = edge->reverse;
					node_j->timestamp = node_i->timestamp;
					node_j->dist_terminal = node_i->dist_terminal + 1;
				}
			}
		}
	}
	else
	{
		// grow sink tree
		for (edge = node_i->first; edge != NULL; edge = edge->next)
		{
			if (edge->reverse->residual_capacity)
			{
				node_j = edge->head;

				if (node_j->parent == NULL)
				{
					node_j->type_tree = SINK;
					node_j->parent = edge->reverse;
					node_j->timestamp = node_i->timestamp;
					node_j->dist_terminal = node_i->dist_terminal + 1;
					set_active(node_j);
				}
				else if (node_j->type_tree == SOURCE)
				{
					edge = edge->reverse;
					break;
				}
				else if (node_j->timestamp <= node_i->timestamp && node_j->dist_terminal > node_i->dist_terminal)
				{
					// heuristic - trying to make the distance from j to the sink shorter
					node_j->parent = edge->reverse;
					node_j->timestamp = node_i->timestamp;
					node_j->dist_terminal = node_i->dist_terminal + 1;
				}
			}
		}
	} // grow sink tree

	if (edge != NULL)
	{
		node_i->next = node_i; // set active flag
		current_node = node_i;
	}
	else
	{
		current_node = NULL;
	}

	return true;
}

void GraphCut::augment_path(GCEdge* connecting_edge)
{
	GCNode* node_i;
	GCEdge* edge;
	float64_t bottleneck;

	// 1. Finding bottleneck capacity
	// 1a the source tree
	bottleneck = connecting_edge->residual_capacity;

	for (node_i = connecting_edge->reverse->head; ; node_i = edge->head)
	{
		edge = node_i->parent;

		if (edge == TERMINAL_EDGE)
		{
			break;
		}

		if (bottleneck > edge->reverse->residual_capacity)
		{
			bottleneck = edge->reverse->residual_capacity;
		}
	}

	if (bottleneck > node_i->tree_cap)
	{
		bottleneck = node_i->tree_cap;
	}

	// 1b the sink tree
	for (node_i = connecting_edge->head; ; node_i = edge->head)
	{
		edge = node_i->parent;

		if (edge == TERMINAL_EDGE)
		{
			break;
		}

		if (bottleneck > edge->residual_capacity)
		{
			bottleneck = edge->residual_capacity;
		}
	}

	if (bottleneck > - node_i->tree_cap)
	{
		bottleneck = - node_i->tree_cap;
	}


	// 2. Augmenting
	// 2a the source tree
	connecting_edge->reverse->residual_capacity += bottleneck;
	connecting_edge->residual_capacity -= bottleneck;

	for (node_i = connecting_edge->reverse->head; ; node_i = edge->head)
	{
		edge = node_i->parent;

		if (edge == TERMINAL_EDGE)
		{
			break;
		}

		edge->residual_capacity += bottleneck;
		edge->reverse->residual_capacity -= bottleneck;

		if (edge->reverse->residual_capacity == 0)
		{
			set_orphan_front(node_i); // add node_i to the beginning of the adoptation list
		}
	}

	node_i->tree_cap -= bottleneck;

	if (node_i->tree_cap == 0)
	{
		set_orphan_front(node_i); // add node_i to the beginning of the adoptation list
	}

	// 2b the sink tree
	for (node_i = connecting_edge->head; ; node_i = edge->head)
	{
		edge = node_i->parent;

		if (edge == TERMINAL_EDGE)
		{
			break;
		}

		edge->reverse->residual_capacity += bottleneck;
		edge->residual_capacity -= bottleneck;

		if (edge->residual_capacity == 0)
		{
			set_orphan_front(node_i);
		}
	}

	node_i->tree_cap += bottleneck;

	if (node_i->tree_cap == 0)
	{
		set_orphan_front(node_i);
	}

	m_flow += bottleneck;
}

void GraphCut::adopt()
{
	GCNodePtr* np, *np_next;
	GCNode* node_i;

	while ((np = m_orphan_first) != NULL)
	{
		np_next = np->next;
		np->next = NULL;

		while ((np = m_orphan_first) != NULL)
		{
			m_orphan_first = np->next;
			node_i = np->ptr;
			SG_FREE(np);

			if (m_orphan_first == NULL)
			{
				m_orphan_last = NULL;
			}

			process_orphan(node_i, node_i->type_tree);
		}

		m_orphan_first = np_next;
	}
}

void GraphCut::set_orphan_front(GCNode* node_i)
{
	GCNodePtr* np;
	node_i->parent = ORPHAN_EDGE;
	np = SG_MALLOC(GCNodePtr, 1);
	np->ptr = node_i;
	np->next = m_orphan_first;
	m_orphan_first = np;
}

void GraphCut::set_orphan_rear(GCNode* node_i)
{
	GCNodePtr* np;
	node_i->parent = ORPHAN_EDGE;
	np = SG_MALLOC(GCNodePtr, 1);
	np->ptr = node_i;

	if (m_orphan_last != NULL)
	{
		m_orphan_last->next = np;
	}
	else
	{
		m_orphan_first = np;
	}

	m_orphan_last = np;
	np->next = NULL;
}

void GraphCut::process_orphan(GCNode* node_i, ETerminalType terminalType_tree)
{
	GCNode* node_j;
	GCEdge* edge0;
	GCEdge* edge0_min = NULL;
	GCEdge* edge;
	int32_t d;
	int32_t d_min = INFINITE_D;

	// trying to find a new parent
	for (edge0 = node_i->first; edge0 != NULL; edge0 = edge0->next)
	{
		if ((terminalType_tree == SOURCE && edge0->reverse->residual_capacity) ||
		        (terminalType_tree == SINK && edge0->residual_capacity))
		{
			node_j = edge0->head;

			if (node_j->type_tree == terminalType_tree && (edge = node_j->parent) != NULL)
			{
				// check the origin of node_j
				d = 0;
				while (1)
				{
					if (node_j->timestamp == m_timestamp)
					{
						d += node_j->dist_terminal;
						break;
					}

					edge = node_j->parent;
					d++;

					if (edge == TERMINAL_EDGE)
					{
						node_j->timestamp = m_timestamp;
						node_j->dist_terminal = 1;
						break;
					}

					if (edge == ORPHAN_EDGE)
					{
						d = INFINITE_D;
						break;
					}

					node_j = edge->head;
				} // while

				if (d < INFINITE_D) // node_j originates from the source, done
				{
					if (d < d_min)
					{
						edge0_min = edge0;
						d_min = d;
					}
					// set marks along the path
					for (node_j = edge0->head; node_j->timestamp != m_timestamp; node_j = node_j->parent->head)
					{
						node_j->timestamp = m_timestamp;
						node_j->dist_terminal = d--;
					}
				}

			} // if node_j->type_tree
		} // if(edge0->reverse->residual_capacity)
	} // for edge0 = node_i->first

	if ((node_i->parent = edge0_min) != NULL)
	{
		node_i->timestamp = m_timestamp;
		node_i->dist_terminal = d_min + 1;
	}
	else
	{
		// no parent is found, process neighbors
		for (edge0 = node_i->first; edge0 != NULL; edge0 = edge0->next)
		{
			node_j = edge0->head;

			if (node_j->type_tree == terminalType_tree && (edge = node_j->parent) != NULL)
			{
				bool is_active_source = (terminalType_tree == SOURCE && edge0->reverse->residual_capacity);
				bool is_active_sink = (terminalType_tree == SINK && edge0->residual_capacity);

				if (is_active_source || is_active_sink)
				{
					set_active(node_j);
				}

				if (edge != TERMINAL_EDGE && edge != ORPHAN_EDGE && edge->head == node_i)
				{
					set_orphan_rear(node_j); // add node_j to the end of the adoptation list
				}
			}
		} // for edge0 = node_i->first
	}
}

ETerminalType GraphCut::get_assignment(int32_t i, ETerminalType default_terminal)
{
	if (m_nodes[i].parent != NULL)
	{
		return m_nodes[i].type_tree;
	}
	else
	{
		return default_terminal;
	}
}

void GraphCut::print_graph()
{
	// print SOURCE-node_i and node_i->SINK edges
	for (int32_t i = 0; i < m_num_nodes; i++)
	{
		GCNode* node_i = m_nodes + i;
		if (node_i->parent == TERMINAL_EDGE)
		{
			if (node_i->type_tree == SOURCE)
			{
				io::print("\n s -> {}, cost = {}", node_i->id, node_i->tree_cap);
			}
			else
			{
				io::print("\n {} -> t, cost = {}", node_i->id, node_i->tree_cap);
			}
		}
	}

	// print node_i->node_j edges
	for (int32_t i = 0; i < m_num_edges; i++)
	{
		GCEdge* edge = m_edges + i;
		io::print("\n {} -> {}, cost = {}", edge->reverse->head->id, edge->head->id, edge->residual_capacity);
	}

}

void GraphCut::print_assignment()
{
	for (int32_t i = 0; i < m_num_nodes; i++)
	{
		GCNode* node_i = m_nodes + i;

		if (get_assignment(i) == SOURCE)
		{
			io::print("\nGCNode {:2d}: S", node_i->id);
		}
		else
		{
			io::print("\nGCNode {:2d}: T", node_i->id);
		}
	}
}

void GraphCut::test_consistency(GCNode* current_node)
{
	GCNode* node_i;
	GCEdge* edge;
	int32_t num1 = 0;
	int32_t num2 = 0;

	// test whether all nodes i with i->next!=NULL are indeed in the queue
	for (int32_t i = 0; i < m_num_nodes; i++)
	{
		node_i = m_nodes + i;
		if (node_i->next || node_i == current_node)
		{
			num1++;
		}
	}

	for (int32_t r = 0; r < 3; r++)
	{
		node_i = (r == 2) ? current_node : m_active_first[r];

		if (node_i)
		{
			for (; ; node_i = node_i->next)
			{
				num2++;

				if (node_i->next == node_i)
				{
					if (r < 2)
						ASSERT(node_i == m_active_last[r])
					else
						ASSERT(node_i == current_node)

					break;
				}
			}
		}
	}

	ASSERT(num1 == num2);

	for (int32_t i = 0; i < m_num_nodes; i++)
	{
		node_i = m_nodes + i;

		// test whether all edges in seach trees are non-saturated
		if (node_i->parent == NULL) {}
		else if (node_i->parent == ORPHAN_EDGE) {}
		else if (node_i->parent == TERMINAL_EDGE)
		{
			if (node_i->type_tree == SOURCE)
				ASSERT(node_i->tree_cap > 0)
			else
				ASSERT(node_i->tree_cap < 0)
		}
		else
		{
			if (node_i->type_tree == SOURCE)
				ASSERT(node_i->parent->reverse->residual_capacity > 0)
			else
				ASSERT(node_i->parent->residual_capacity > 0)
		}

		// test whether passive nodes in search trees have neighbors in
		// a different tree through non-saturated edges
		if (node_i->parent && !node_i->next)
		{
			if (node_i->type_tree == SOURCE)
			{
				ASSERT(node_i->tree_cap >= 0);

				for (edge = node_i->first; edge; edge = edge->next)
				{
					if (edge->residual_capacity > 0)
					{
						ASSERT(edge->head->parent && edge->head->type_tree == SOURCE);
					}
				}
			}
			else
			{
				ASSERT(node_i->tree_cap <= 0);

				for (edge = node_i->first; edge; edge = edge->next)
				{
					if (edge->reverse->residual_capacity > 0)
					{
						ASSERT(edge->head->parent && (edge->head->type_tree == SINK));
					}
				}
			}
		}
		// test marking invariants
		if (node_i->parent && node_i->parent != ORPHAN_EDGE && node_i->parent != TERMINAL_EDGE)
		{
			ASSERT(node_i->timestamp <= node_i->parent->head->timestamp);

			if (node_i->timestamp == node_i->parent->head->timestamp)
			{
				ASSERT(node_i->dist_terminal > node_i->parent->head->dist_terminal);
			}
		}
	}
}
