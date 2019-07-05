/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evgeniy Andreev,
 *          Sergey Lisitsyn, Soumyajit De, Shashwat Lal Das, Fernando Iglesias,
 *          Bjoern Esser, Wu Lin
 */

#include <shogun/features/Features.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>

#include <string.h>

using namespace shogun;

Features::Features(int32_t size)
: SGObject()
{
	init();
	cache_size = size;
}

Features::Features(const Features& orig)
: SGObject(orig)
{
	init();

	// Call to init creates new preproc arrays.
	// FIXME: make_clone?
	preproc = orig.preproc;
}

Features::Features(std::shared_ptr<File> loader)
: SGObject()
{
	init();

	load(loader);
	io::info("Feature object loaded ({})",fmt::ptr(this));
}

Features::~Features()
{
	clean_preprocessors();
}

void Features::init()
{
	set_default_mask(ParameterProperties::READONLY);

	SG_ADD(&properties, "properties", "Feature properties");
	SG_ADD(&cache_size, "cache_size", "Size of cache in MB");

	SG_ADD(&preproc, "preproc", "Array of preprocessors.");

	SG_ADD((std::shared_ptr<SGObject>*)&m_subset_stack, "subset_stack", "Stack of subsets");

	m_subset_stack=std::make_shared<SubsetStack>();


	properties = FP_NONE;
	cache_size = 0;
}

void Features::add_preprocessor(std::shared_ptr<Preprocessor> p)
{
	ASSERT(p)

	preproc.push_back(p);
}

std::shared_ptr<Preprocessor> Features::get_preprocessor(int32_t num) const
{
	if (num<preproc.size() && num>=0)
	{
	  return preproc[num];
	}
	else
		return NULL;
}

void Features::clean_preprocessors()
{
	preproc.clear();
}

void Features::del_preprocessor(int32_t num)
{
	if (num<preproc.size() && num>=0)
	{
		preproc.erase(preproc.cbegin()+num);
	}
}

void Features::list_preprocessors()
{
	index_t i = 0;
	for (const auto& v: preproc)
	{
		io::info("preproc[{}]={}\n", i++, v->get_name());
	}
}

int32_t Features::get_num_preprocessors() const
{
	return preproc.size();
}

int32_t Features::get_cache_size() const
{
	return cache_size;
}

bool Features::reshape(int32_t num_features, int32_t num_vectors)
{
	not_implemented(SOURCE_LOCATION);
	return false;
}

void Features::load(std::shared_ptr<File> loader)
{
	SG_SET_LOCALE_C;
	not_implemented(SOURCE_LOCATION);
	SG_RESET_LOCALE;
}

void Features::save(std::shared_ptr<File> writer)
{
	SG_SET_LOCALE_C;
	not_implemented(SOURCE_LOCATION);
	SG_RESET_LOCALE;
}

bool Features::check_feature_compatibility(std::shared_ptr<Features> f) const
{
	bool result=false;

	if (f)
	{
		result= ( (this->get_feature_class() == f->get_feature_class()) &&
				(this->get_feature_type() == f->get_feature_type()));
	}
	return result;
}

bool Features::has_property(EFeatureProperty p) const
{
	return (properties & p) != 0;
}

void Features::set_property(EFeatureProperty p)
{
	properties |= p;
}

void Features::unset_property(EFeatureProperty p)
{
	properties &= (properties | p) ^ p;
}

void Features::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
	subset_changed_post();
}

void Features::add_subset_in_place(SGVector<index_t> subset)
{
	m_subset_stack->add_subset_in_place(subset);
	subset_changed_post();
}

void Features::remove_subset()
{
	m_subset_stack->remove_subset();
	subset_changed_post();
}

void Features::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
	subset_changed_post();
}

std::shared_ptr<SubsetStack> Features::get_subset_stack()
{

	return m_subset_stack;
}

std::shared_ptr<Features> Features::copy_subset(SGVector<index_t> indices) const
{
	error("{}::copy_subset(): copy_subset and therefore model storage of "
			"Machine (required for cross-validation and model-selection is "
			"not yet implemented yet. Ask developers!", get_name());
	return NULL;
}

std::shared_ptr<Features> Features::copy_dimension_subset(SGVector<index_t> dims) const
{
	io::warn("{}::copy_dimension_subset():: Is not yet implemented!",
			get_name());
	return NULL;
}

bool Features::get_feature_class_compatibility(EFeatureClass rhs) const
{
	if (this->get_feature_class()==rhs)
		return true;
	return false;
}
