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
#include <shogun/lib/DynamicObjectArray.h>

#include <string.h>

using namespace shogun;

CFeatures::CFeatures(int32_t size)
: CSGObject()
{
	init();
	cache_size = size;
}

CFeatures::CFeatures(const CFeatures& orig)
: CSGObject(orig)
{
	init();

	// TODO this should be a shallow copy
	auto old_preproc = preproc;
	preproc = make_clone(preproc);
	SG_UNREF(old_preproc);
}

CFeatures::CFeatures(CFile* loader)
: CSGObject()
{
	init();

	load(loader);
	io::info("Feature object loaded ({})\n",fmt::ptr(this));
}

CFeatures::~CFeatures()
{
	clean_preprocessors();
	SG_UNREF(m_subset_stack);
	SG_UNREF(preproc);
}

void CFeatures::init()
{
	set_default_mask(ParameterProperties::READONLY);

	SG_ADD(&properties, "properties", "Feature properties");
	SG_ADD(&cache_size, "cache_size", "Size of cache in MB");

	SG_ADD((CSGObject**) &preproc, "preproc", "Array of preprocessors.");

	SG_ADD((CSGObject**)&m_subset_stack, "subset_stack", "Stack of subsets");

	m_subset_stack=new CSubsetStack();
	SG_REF(m_subset_stack);

	properties = FP_NONE;
	cache_size = 0;
	preproc = new CDynamicObjectArray();
	SG_REF(preproc);
}

void CFeatures::add_preprocessor(CPreprocessor* p)
{
	ASSERT(p)

	preproc->push_back(p);
}

CPreprocessor* CFeatures::get_preprocessor(int32_t num) const
{
	if (num<preproc->get_num_elements() && num>=0)
	{
	  return (CPreprocessor*) preproc->get_element(num);
	}
	else
		return NULL;
}

void CFeatures::clean_preprocessors()
{
	preproc->reset_array();
}

void CFeatures::del_preprocessor(int32_t num)
{
	if (num<preproc->get_num_elements() && num>=0)
	{
		preproc->delete_element(num);
	}
}

void CFeatures::list_preprocessors()
{
	int32_t num_preproc = preproc->get_num_elements();

	for (int32_t i=0; i<num_preproc; i++)
	{
		io::info("preproc[{}]={}\n", i, preproc->get_element(i)->get_name());
	}
}

int32_t CFeatures::get_num_preprocessors() const
{
	return preproc->get_num_elements();
}

int32_t CFeatures::get_cache_size() const
{
	return cache_size;
}

bool CFeatures::reshape(int32_t num_features, int32_t num_vectors)
{
	not_implemented(SOURCE_LOCATION);
	return false;
}

void CFeatures::list_feature_obj() const
{
	io::info("{} - ", fmt::ptr(this));
	switch (get_feature_class())
	{
		case C_UNKNOWN:
			io::info("C_UNKNOWN ");
			break;
		case C_DENSE:
			io::info("C_DENSE ");
			break;
		case C_SPARSE:
			io::info("C_SPARSE ");
			break;
		case C_STRING:
			io::info("C_STRING ");
			break;
		case C_COMBINED:
			io::info("C_COMBINED ");
			break;
		case C_COMBINED_DOT:
			io::info("C_COMBINED_DOT ");
			break;
		case C_WD:
			io::info("C_WD ");
			break;
		case C_SPEC:
			io::info("C_SPEC ");
			break;
		case C_WEIGHTEDSPEC:
			io::info("C_WEIGHTEDSPEC ");
			break;
		case C_STREAMING_DENSE:
			io::info("C_STREAMING_DENSE ");
			break;
		case C_STREAMING_SPARSE:
			io::info("C_STREAMING_SPARSE ");
			break;
		case C_STREAMING_STRING:
			io::info("C_STREAMING_STRING ");
			break;
		case C_STREAMING_VW:
			io::info("C_STREAMING_VW ");
			break;
		case C_ANY:
			io::info("C_ANY ");
			break;
		default:
         error("ERROR UNKNOWN FEATURE CLASS");
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			io::info("F_UNKNOWN \n");
			break;
		case F_CHAR:
			io::info("F_CHAR \n");
			break;
		case F_BYTE:
			io::info("F_BYTE \n");
			break;
		case F_SHORT:
			io::info("F_SHORT \n");
			break;
		case F_WORD:
			io::info("F_WORD \n");
			break;
		case F_INT:
			io::info("F_INT \n");
			break;
		case F_UINT:
			io::info("F_UINT \n");
			break;
		case F_LONG:
			io::info("F_LONG \n");
			break;
		case F_ULONG:
			io::info("F_ULONG \n");
			break;
		case F_SHORTREAL:
			io::info("F_SHORTEAL \n");
			break;
		case F_DREAL:
			io::info("F_DREAL \n");
			break;
		case F_LONGREAL:
			io::info("F_LONGREAL \n");
			break;
		case F_ANY:
			io::info("F_ANY \n");
			break;
		default:
         error("ERROR UNKNOWN FEATURE TYPE\n");
	}
}


void CFeatures::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	not_implemented(SOURCE_LOCATION);
	SG_RESET_LOCALE;
}

void CFeatures::save(CFile* writer)
{
	SG_SET_LOCALE_C;
	not_implemented(SOURCE_LOCATION);
	SG_RESET_LOCALE;
}

bool CFeatures::check_feature_compatibility(CFeatures* f) const
{
	bool result=false;

	if (f)
	{
		result= ( (this->get_feature_class() == f->get_feature_class()) &&
				(this->get_feature_type() == f->get_feature_type()));
	}
	return result;
}

bool CFeatures::has_property(EFeatureProperty p) const
{
	return (properties & p) != 0;
}

void CFeatures::set_property(EFeatureProperty p)
{
	properties |= p;
}

void CFeatures::unset_property(EFeatureProperty p)
{
	properties &= (properties | p) ^ p;
}

void CFeatures::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
	subset_changed_post();
}

void CFeatures::add_subset_in_place(SGVector<index_t> subset)
{
	m_subset_stack->add_subset_in_place(subset);
	subset_changed_post();
}

void CFeatures::remove_subset()
{
	m_subset_stack->remove_subset();
	subset_changed_post();
}

void CFeatures::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
	subset_changed_post();
}

CSubsetStack* CFeatures::get_subset_stack()
{
	SG_REF(m_subset_stack);
	return m_subset_stack;
}

CFeatures* CFeatures::copy_subset(SGVector<index_t> indices) const
{
	error("{}::copy_subset(): copy_subset and therefore model storage of "
			"CMachine (required for cross-validation and model-selection is "
			"not yet implemented yet. Ask developers!\n", get_name());
	return NULL;
}

CFeatures* CFeatures::copy_dimension_subset(SGVector<index_t> dims) const
{
	io::warn("{}::copy_dimension_subset():: Is not yet implemented!\n",
			get_name());
	return NULL;
}

bool CFeatures::get_feature_class_compatibility(EFeatureClass rhs) const
{
	if (this->get_feature_class()==rhs)
		return true;
	return false;
}
