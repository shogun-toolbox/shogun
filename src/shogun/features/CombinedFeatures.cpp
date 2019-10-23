/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evangelos Anagnostopoulos,
 *          Vladislav Horbatiuk, Evgeniy Andreev, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/features/CombinedFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Set.h>
#include <shogun/lib/Map.h>

using namespace shogun;

CCombinedFeatures::CCombinedFeatures()
: CFeatures(0)
{
	init();
	num_vec=0;
}

CCombinedFeatures::CCombinedFeatures(const CCombinedFeatures& orig)
: CFeatures(orig)
{
	init();

	// appending below uses this, mem error otherwise if not done first
	num_vec=orig.num_vec;

	for (auto i : range(orig.get_num_feature_obj()))
	{
		auto f = orig.get_feature_obj(i);
		append_feature_obj(f->duplicate());
		SG_UNREF(f);
	}

	if (orig.m_subset_stack)
	{
		auto old = m_subset_stack;
		m_subset_stack=new CSubsetStack(*orig.m_subset_stack);
		SG_REF(m_subset_stack);
		SG_UNREF(old);
	}
}

CFeatures* CCombinedFeatures::duplicate() const
{
	return new CCombinedFeatures(*this);
}

CCombinedFeatures::~CCombinedFeatures()
{
	SG_UNREF(feature_array);
}

CFeatures* CCombinedFeatures::get_feature_obj(int32_t idx) const
{
	require(
	    idx < get_num_feature_obj() && idx>=0, "Feature index ({}) must be within [{}, {}]",
	    idx, 0, get_num_feature_obj()-1);
	return (CFeatures*) feature_array->get_element(idx);
}

bool CCombinedFeatures::check_feature_obj_compatibility(CCombinedFeatures* comb_feat)
{
	bool result=false;

	if ( (comb_feat) && (this->get_num_feature_obj() == comb_feat->get_num_feature_obj()) )
	{
		for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
		{
			CFeatures* f1=this->get_feature_obj(f_idx);
			CFeatures* f2=comb_feat->get_feature_obj(f_idx);

			if ( ! (f1 && f2 && f1->check_feature_compatibility(f2)) )
			{
				SG_UNREF(f1);
				SG_UNREF(f2);
				io::info("not compatible, combfeat");
				io::info("{}", comb_feat->to_string().c_str());
				io::info("vs this");
				io::info("{}", this->to_string().c_str());
				return false;
			}

			SG_UNREF(f1);
			SG_UNREF(f2);
		}
		SG_DEBUG("features are compatible")
		result=true;
	}
	else
	{
		if (!comb_feat)
		{
			io::warn("comb_feat is NULL ");
		}
		else
		{
			io::warn("number of features in combined feature objects differs ({} != {})", this->get_num_feature_obj(), comb_feat->get_num_feature_obj());
				io::info("compare");
				io::info("{}", comb_feat->to_string().c_str());
			io::info("vs this");
				io::info("{}", this->to_string().c_str());
		}
	}

	return result;
}

CFeatures* CCombinedFeatures::get_first_feature_obj() const
{
	return get_feature_obj(0);
}

CFeatures* CCombinedFeatures::get_last_feature_obj() const
{
	return get_feature_obj(get_num_feature_obj()-1);
}

bool CCombinedFeatures::insert_feature_obj(CFeatures* obj, int32_t idx)
{
	ASSERT(obj)
	int32_t n=obj->get_num_vectors();

	if (get_num_vectors()>0 && n!=get_num_vectors())
	{
		error("Number of feature vectors does not match (expected {}, "
				"obj has {})", get_num_vectors(), n);
	}

	num_vec=n;
	return feature_array->insert_element(obj, idx);
}

bool CCombinedFeatures::append_feature_obj(CFeatures* obj)
{
	ASSERT(obj)
	int32_t n=obj->get_num_vectors();

	if (get_num_vectors()>0 && n!=get_num_vectors())
	{
		error("Number of feature vectors does not match (expected {}, "
				"obj has {})", get_num_vectors(), n);
	}

	num_vec=n;

	int num_feature_obj = get_num_feature_obj();
	feature_array->push_back(obj);
	return num_feature_obj+1 == feature_array->get_num_elements();
}

bool CCombinedFeatures::delete_feature_obj(int32_t idx)
{
	return feature_array->delete_element(idx);
}

int32_t CCombinedFeatures::get_num_feature_obj() const
{
	return feature_array->get_num_elements();
}

void CCombinedFeatures::init()
{
	feature_array = new CDynamicObjectArray();
	SG_REF(feature_array);

	SG_ADD(&num_vec, "num_vec", "Number of vectors.");
	SG_ADD(&feature_array, "feature_array", "Feature array.");
}

CFeatures* CCombinedFeatures::create_merged_copy(CFeatures* other) const
{
	/* TODO, if all features are the same, only one copy should be created
	 * in memory */
	SG_TRACE("entering {}::create_merged_copy()", get_name());
	if (get_feature_type()!=other->get_feature_type() ||
			get_feature_class()!=other->get_feature_class() ||
			strcmp(get_name(), other->get_name()))
	{
		error("{}::create_merged_copy(): Features are of different type!",
				get_name());
	}

	auto casted = dynamic_cast<const CCombinedFeatures*>(other);

	if (!casted)
	{
		error("{}::create_merged_copy(): Could not cast object of {} to "
				"same type as {}\n",get_name(), other->get_name(), get_name());
	}

	if (get_num_feature_obj()!=casted->get_num_feature_obj())
	{
		error("{}::create_merged_copy(): Only possible if both instances "
				"have the same number of sub-feature-objects\n", get_name());
	}

	CCombinedFeatures* result=new CCombinedFeatures();
	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current_this=get_feature_obj(f_idx);
		CFeatures* current_other=casted->get_feature_obj(f_idx);

		result->append_feature_obj(
				current_this->create_merged_copy(current_other));
		SG_UNREF(current_this);
		SG_UNREF(current_other);
	}

	SG_TRACE("leaving {}::create_merged_copy()", get_name());
	return result;
}

void CCombinedFeatures::add_subset(SGVector<index_t> subset)
{
	SG_TRACE("entering {}::add_subset()", get_name());
	CSet<CFeatures*>* processed=new CSet<CFeatures*>();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);

		if (!processed->contains(current))
		{
			/* remember that subset was added here */
			current->add_subset(subset);
			processed->add(current);
			SG_DEBUG("adding subset to {} at {}",
					current->get_name(), fmt::ptr(current));
		}
		SG_UNREF(current);
	}

	/* also add subset to local stack to have it for easy access */
	m_subset_stack->add_subset(subset);

	subset_changed_post();
	SG_UNREF(processed);
	SG_TRACE("leaving {}::add_subset()", get_name());
}

void CCombinedFeatures::remove_subset()
{
	SG_TRACE("entering {}::remove_subset()", get_name());
	CSet<CFeatures*>* processed=new CSet<CFeatures*>();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);
		if (!processed->contains(current))
		{
			/* remember that subset was added here */
			current->remove_subset();
			processed->add(current);
			SG_DEBUG("removing subset from {} at {}",
					current->get_name(), fmt::ptr(current));
		}
		SG_UNREF(current);
	}

	/* also remove subset from local stack to have it for easy access */
	m_subset_stack->remove_subset();

	subset_changed_post();
	SG_UNREF(processed);
	SG_TRACE("leaving {}::remove_subset()", get_name());
}

void CCombinedFeatures::remove_all_subsets()
{
	SG_TRACE("entering {}::remove_all_subsets()", get_name());
	CSet<CFeatures*>* processed=new CSet<CFeatures*>();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);
		if (!processed->contains(current))
		{
			/* remember that subset was added here */
			current->remove_all_subsets();
			processed->add(current);
			SG_DEBUG("removing all subsets from {} at {}",
					current->get_name(), fmt::ptr(current));
		}
		SG_UNREF(current);
	}

	/* also remove subsets from local stack to have it for easy access */
	m_subset_stack->remove_all_subsets();

	subset_changed_post();
	SG_UNREF(processed);
	SG_TRACE("leaving {}::remove_all_subsets()", get_name());
}

CFeatures* CCombinedFeatures::copy_subset(SGVector<index_t> indices) const
{
	/* this is returned with the results of copy_subset of sub-features */
	CCombinedFeatures* result=new CCombinedFeatures();

	/* map to only copy same feature objects once */
	CMap<CFeatures*, CFeatures*>* processed=new CMap<CFeatures*, CFeatures*>();
	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);

		CFeatures* new_element=NULL;

		/* only copy if not done yet, otherwise, use old copy */
		if (!processed->contains(current))
		{
			new_element=current->copy_subset(indices);
			processed->add(current, new_element);
		}
		else
		{
			new_element=processed->get_element(current);

			/* has to be SG_REF'ed since it will be unrefed afterwards */
			SG_REF(new_element);
		}

		/* add to result */
		result->append_feature_obj(new_element);

		/* clean up: copy_subset of SG_REF has to be undone */
		SG_UNREF(new_element);

		SG_UNREF(current);
	}

	SG_UNREF(processed);

	SG_REF(result);
	return result;
}
