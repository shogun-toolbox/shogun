/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Abinash Panda, Viktor Gal, Bjoern Esser, Sergey Lisitsyn,
 *          Jiaolong Xu, Sanuj Sharma
 */

#include <shogun/structure/FactorGraphModel.h>
#include <shogun/structure/Factor.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <unordered_map>
#include <utility>
typedef std::unordered_map<int32_t, int32_t> factor_counts_type;

using namespace shogun;

FactorGraphModel::FactorGraphModel()
	: StructuredModel()
{
	init();
}

FactorGraphModel::FactorGraphModel(std::shared_ptr<Features> features, std::shared_ptr<StructuredLabels> labels,
	EMAPInferType inf_type, bool verbose) : StructuredModel(std::move(features), std::move(labels))
{
	init();
	m_inf_type = inf_type;
	m_verbose = verbose;
}

FactorGraphModel::~FactorGraphModel()
{

}

void FactorGraphModel::init()
{
	SG_ADD(&m_factor_types, "factor_types", "Array of factor types");
	SG_ADD(&m_w_cache, "w_cache", "Cache of global parameters");
	SG_ADD(&m_w_map, "w_map", "Parameter mapping");

	m_inf_type = TREE_MAX_PROD;
	m_factor_types.clear();
	m_verbose = false;


}

void FactorGraphModel::add_factor_type(const std::shared_ptr<FactorType>& ftype)
{
	require(ftype->get_w_dim() > 0, "{}::add_factor_type(): number of parameters can't be 0!",
		get_name());

	// check whether this ftype has been added
	int32_t id = ftype->get_type_id();
	for (auto& ft : m_factor_types)
	{
		if (id == ft->get_type_id())
		{
			io::print("{}::add_factor_type(): factor_type (id = {}) has been added!\n",
				get_name(), id);

			return;
		}


	}

	SGVector<int32_t> w_map_cp = m_w_map.clone();
	m_w_map.resize_vector(w_map_cp.size() + ftype->get_w_dim());

	for (int32_t mi = 0; mi < w_map_cp.size(); mi++)
	{
		m_w_map[mi] = w_map_cp[mi];
	}
	// add new mapping in the end
	for (int32_t mi = w_map_cp.size(); mi < m_w_map.size(); mi++)
	{
		m_w_map[mi] = id;
	}

	// push factor type
	m_factor_types.push_back(ftype);

	// initialize w cache
	fparams_to_w();

	if (m_verbose)
	{
		m_w_map.display_vector("add_factor_type(): m_w_map");
	}
}

void FactorGraphModel::del_factor_type(const int32_t ftype_id)
{
	int w_dim = 0;
	// delete from m_factor_types
	for (auto fi = m_factor_types.begin(); fi != m_factor_types.end(); ++fi)
	{
		auto ftype = *fi;
		if (ftype_id == ftype->get_type_id())
		{
			w_dim = ftype->get_w_dim();

			m_factor_types.erase(fi);
			break;
		}


	}

	ASSERT(w_dim != 0);

	SGVector<int32_t> w_map_cp = m_w_map.clone();
	m_w_map.resize_vector(w_map_cp.size() - w_dim);

	int ind = 0;
	for (int32_t mi = 0; mi < w_map_cp.size(); mi++)
	{
		if (w_map_cp[mi] == ftype_id)
			continue;

		m_w_map[ind++] = w_map_cp[mi];
	}

	ASSERT(ind == m_w_map.size());
}

std::vector<std::shared_ptr<FactorType>> FactorGraphModel::get_factor_types() const
{

	return m_factor_types;
}

std::shared_ptr<FactorType> FactorGraphModel::get_factor_type(const int32_t ftype_id) const
{
	for (auto& ftype : m_factor_types)
	{
		if (ftype_id == ftype->get_type_id())
			return ftype;


	}

	return NULL;
}

SGVector<int32_t> FactorGraphModel::get_global_params_mapping() const
{
	return m_w_map.clone();
}

SGVector<int32_t> FactorGraphModel::get_params_mapping(const int32_t ftype_id)
{
	return m_w_map.find(ftype_id);
}

int32_t FactorGraphModel::get_dim() const
{
	return m_w_map.size();
}

SGVector<float64_t> FactorGraphModel::fparams_to_w()
{
	if (m_w_cache.size() != get_dim())
		m_w_cache.resize_vector(get_dim());

	int32_t offset = 0;
	for (auto& ftype : m_factor_types)
	{
		int32_t w_dim = ftype->get_w_dim();
		offset += w_dim;
		SGVector<float64_t> fw = ftype->get_w();
		SGVector<int32_t> fw_map = get_params_mapping(ftype->get_type_id());

		ASSERT(fw_map.size() == fw.size());

		for (int32_t wi = 0; wi < w_dim; wi++)
			m_w_cache[fw_map[wi]] = fw[wi];


	}

	ASSERT(offset == m_w_cache.size());

	return m_w_cache;
}

void FactorGraphModel::w_to_fparams(SGVector<float64_t> w)
{
	// if nothing changed
	if (m_w_cache.equals(w))
		return;

	if (m_verbose)
		io::print("****** update m_w_cache!\n");

	ASSERT(w.size() == m_w_cache.size());
	m_w_cache = w.clone();

	int32_t offset = 0;
	for (auto& ftype : m_factor_types)
	{
		int32_t w_dim = ftype->get_w_dim();
		offset += w_dim;
		SGVector<float64_t> fw(w_dim);
		SGVector<int32_t> fw_map = get_params_mapping(ftype->get_type_id());

		for (int32_t wi = 0; wi < w_dim; wi++)
			fw[wi] = m_w_cache[fw_map[wi]];

		ftype->set_w(fw);

	}

	ASSERT(offset == m_w_cache.size());
}

SGVector< float64_t > FactorGraphModel::get_joint_feature_vector(int32_t feat_idx, std::shared_ptr<StructuredData> y)
{
	// factor graph instance
	auto mf = m_features->as<FactorGraphFeatures>();
	auto fg = mf->get_sample(feat_idx);

	// ground truth states
	auto fg_states = y->as<FactorGraphObservation>();
	SGVector<int32_t> states = fg_states->get_data();

	// initialize psi
	SGVector<float64_t> psi(get_dim());
	psi.zero();

	// construct unnormalized psi
	auto facs = fg->get_factors();
	for (auto& fac : facs)
	{
		auto ftype = fac->get_factor_type();
		int32_t id = ftype->get_type_id();
		SGVector<int32_t> w_map = get_params_mapping(id);

		ASSERT(w_map.size() == ftype->get_w_dim());

		SGVector<float64_t> dat = fac->get_data();
		int32_t dat_size = dat.size();

		ASSERT(w_map.size() == dat_size * ftype->get_num_assignments());

		int32_t ei = ftype->index_from_universe_assignment(states, fac->get_variables());
		for (int32_t di = 0; di < dat_size; di++)
			psi[w_map[ei*dat_size + di]] += dat[di];



	}

	// negation (-E(x,y) = <w,phi(x,y)>)
	psi.scale(-1.0);


	return psi;
}

// E(x_i, y; w) - E(x_i, y_i; w) >= L(y_i, y) - xi_i
// xi_i >= max oracle
// max oracle := argmax_y { L(y_i, y) - E(x_i, y; w) + E(x_i, y_i; w) }
//            := argmin_y { -L(y_i, y) + E(x_i, y; w) } - E(x_i, y_i; w)
// we do energy minimization in inference, so get back to max oracle value is:
// [ L(y_i, y_star) - E(x_i, y_star; w) ] + E(x_i, y_i; w)
std::shared_ptr<ResultSet> FactorGraphModel::argmax(SGVector<float64_t> w, int32_t feat_idx, bool const training)
{
	// factor graph instance
	auto mf = m_features->as<FactorGraphFeatures>();
	auto fg = mf->get_sample(feat_idx);

	// prepare factor graph
	fg->connect_components();
	if (m_inf_type == TREE_MAX_PROD)
	{
		ASSERT(fg->is_tree_graph());
	}

	if (m_verbose)
		io::print("\n------ example {}\n", feat_idx);

	// update factor parameters
	w_to_fparams(w);
	fg->compute_energies();

	if (m_verbose)
	{
		io::print("energy table before loss-aug:\n");
		fg->evaluate_energies();
	}

	// prepare ResultSet
	auto ret = std::make_shared<ResultSet>();

	ret->psi_computed = true;

	// y_truth
	auto y_truth = m_labels->get_label(feat_idx)->as<FactorGraphObservation>();

	SGVector<int32_t> states_gt = y_truth->get_data();

	// E(x_i, y_i; w)
	ret->psi_truth = get_joint_feature_vector(feat_idx, y_truth);
	float64_t energy_gt = fg->evaluate_energy(states_gt);
	ret->score = energy_gt;

	// - min_y [ E(x_i, y; w) - delta(y_i, y) ]
	if (training)
	{
		fg->loss_augmentation(y_truth); // wrong assignments -delta()

		if (m_verbose)
		{
			io::print("energy table after loss-aug:\n");
			fg->evaluate_energies();
		}
	}

	MAPInference infer_met(fg, m_inf_type);
	infer_met.inference();

	// y_star
	auto y_star = infer_met.get_structured_outputs();
	SGVector<int32_t> states_star = y_star->get_data();

	ret->argmax = y_star;
	ret->psi_pred = get_joint_feature_vector(feat_idx, y_star);
	float64_t l_energy_pred = fg->evaluate_energy(states_star);
	ret->score -= l_energy_pred;
	ret->delta = delta_loss(y_truth, y_star);

	if (m_verbose)
	{
		float64_t dot_pred = linalg::dot(w, ret->psi_pred);
		float64_t dot_truth = linalg::dot(w, ret->psi_truth);
		float64_t slack =  dot_pred + ret->delta - dot_truth;

		io::print("\n");
		w.display_vector("w");

		ret->psi_pred.display_vector("psi_pred");
		states_star.display_vector("state_pred");

		io::print("dot_pred = {}, energy_pred = {}, delta = {}\n\n", dot_pred, l_energy_pred, ret->delta);

		ret->psi_truth.display_vector("psi_truth");
		states_gt.display_vector("state_gt");

		io::print("dot_truth = {}, energy_gt = {}\n\n", dot_truth, energy_gt);

		io::print("slack = {}, score = {}\n\n", slack, ret->score);
	}


	return ret;
}

float64_t FactorGraphModel::delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2)
{
	auto y_truth = y1->as<FactorGraphObservation>();
	auto y_pred = y2->as<FactorGraphObservation>();
	SGVector<int32_t> s_truth = y_truth->get_data();
	SGVector<int32_t> s_pred = y_pred->get_data();

	ASSERT(s_pred.size() == s_truth.size());

	float64_t loss = 0.0;
	for (int32_t si = 0; si < s_pred.size(); si++)
	{
		if (s_pred[si] != s_truth[si])
			loss += y_truth->get_loss_weights()[si];
	}

	return loss;
}

void FactorGraphModel::init_training()
{
}

void FactorGraphModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > & lb,
		SGVector< float64_t > & ub,
		SGMatrix< float64_t > & C)
{
	C = SGMatrix< float64_t >::create_identity_matrix(get_dim(), regularization);

	int32_t dim_w = get_dim();

	switch (m_inf_type)
	{
		case GRAPH_CUT:
			lb.resize_vector(dim_w);
			ub.resize_vector(dim_w);
			SGVector< float64_t >::fill_vector(lb.vector, lb.vlen, -Math::INFTY);
			SGVector< float64_t >::fill_vector(ub.vector, ub.vlen, Math::INFTY);

			for (auto& ftype : m_factor_types)
			{
				int32_t w_dim = ftype->get_w_dim();
				SGVector<int32_t> card = ftype->get_cardinalities();

				// TODO: Features of pairwise factor are assume to be 1. Consider more general case, e.g., edge features are availabel.
				// for pairwise factors with binary labels
				if (card.size() == 2 &&  card[0] == 2 && card[1] == 2)
				{
					require(w_dim == 4, "GraphCut doesn't support edge features currently.");
					SGVector<float64_t> fw = ftype->get_w();
					SGVector<int32_t> fw_map = get_params_mapping(ftype->get_type_id());
					ASSERT(fw_map.size() == fw.size());

					// submodularity constrain
					// E(0,1) + E(1,0) - E(0,0) + E(1,1) > 0
					// For pairwise factors, data term = 1,
					// energy table indeces are defined as follows:
					// w[0]*1 = E(0, 0)
					// w[1]*1 = E(1, 0)
					// w[2]*1 = E(0, 1)
					// w[3]*1 = E(1, 1)
					// thus, w[2] + w[1] - w[0] - w[3] > 0
					// since factor graph model is over-parametering,
					// the constrain can be written as w[2] > 0, w[1] > 0, w[0] = 0, w[3] = 0
					lb[fw_map[0]] = 0;
					ub[fw_map[0]] = 0;
					lb[fw_map[3]] = 0;
					ub[fw_map[3]] = 0;
					lb[fw_map[1]] = 0;
					lb[fw_map[2]] = 0;
				}

			}
			break;
		case TREE_MAX_PROD:
		case LOOPY_MAX_PROD:
		case LP_RELAXATION:
		case TRWS_MAX_PROD:
		case GEMP_LP:
			break;
	}
}
