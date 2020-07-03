/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evangelos Anagnostopoulos,
 *          Vladislav Horbatiuk, Evgeniy Andreev, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/features/CombinedFeatures.h>
#include <shogun/io/SGIO.h>

#include <unordered_map>
#include <unordered_set>

using namespace shogun;

CombinedFeatures::CombinedFeatures()
: Features(0)
{
	init();
	num_vec=0;
}

CombinedFeatures::CombinedFeatures(const CombinedFeatures& orig)
: Features(orig)
{
	init();

	// appending below uses this, mem error otherwise if not done first
	num_vec=orig.num_vec;

	for (auto i : range(orig.get_num_feature_obj()))
	{
		auto f = orig.get_feature_obj(i);
		append_feature_obj(f->duplicate());
	}

	if (orig.m_subset_stack)
	{
		m_subset_stack=std::make_shared<SubsetStack>(*orig.m_subset_stack);
	}
}

std::shared_ptr<Features> CombinedFeatures::duplicate() const
{
	return std::make_shared<CombinedFeatures>(*this);
}

CombinedFeatures::~CombinedFeatures()
{

}

std::shared_ptr<Features> CombinedFeatures::get_feature_obj(int32_t idx) const
{
	require(
	    idx < get_num_feature_obj() && idx>=0, "Feature index ({}) must be within [{}, {}]",
	    idx, 0, get_num_feature_obj()-1);
	return feature_array[idx];
}

bool CombinedFeatures::check_feature_obj_compatibility(const std::shared_ptr<CombinedFeatures>& comb_feat)
{
	bool result=false;

	if ( (comb_feat) && (this->get_num_feature_obj() == comb_feat->get_num_feature_obj()) )
	{
		for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
		{
			auto f1=this->get_feature_obj(f_idx);
			auto f2=comb_feat->get_feature_obj(f_idx);

			if ( ! (f1 && f2 && f1->check_feature_compatibility(f2)) )
			{
				io::info("not compatible, combfeat");
				io::info("{}", comb_feat->to_string().c_str());
				io::info("vs this");
				io::info("{}", this->to_string().c_str());
				return false;
			}



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

std::shared_ptr<Features> CombinedFeatures::get_first_feature_obj() const
{
	return get_feature_obj(0);
}

std::shared_ptr<Features> CombinedFeatures::get_last_feature_obj() const
{
	return get_feature_obj(get_num_feature_obj()-1);
}

bool CombinedFeatures::insert_feature_obj(const std::shared_ptr<Features>& obj, int32_t idx)
{
	ASSERT(obj)
	int32_t n=obj->get_num_vectors();

	if (get_num_vectors()>0 && n!=get_num_vectors())
	{
		error("Number of feature vectors does not match (expected {}, "
				"obj has {})", get_num_vectors(), n);
	}

	num_vec=n;
	feature_array.insert(feature_array.begin()+idx, obj);
	return true;
}

bool CombinedFeatures::append_feature_obj(std::shared_ptr<Features> obj)
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
	feature_array.push_back(std::move(obj));
	return num_feature_obj+1 == feature_array.size();
}

bool CombinedFeatures::delete_feature_obj(int32_t idx)
{
	feature_array.erase(feature_array.cbegin()+idx);
	return true;
}

int32_t CombinedFeatures::get_num_feature_obj() const
{
	return feature_array.size();
}

void CombinedFeatures::init()
{
	SG_ADD(&num_vec, "num_vec", "Number of vectors.");
	SG_ADD(&feature_array, "feature_array", "Feature array.");
	add_callback_function("feature_array", [this](){
		int32_t n=feature_array.back()->get_num_vectors();
		if (get_num_vectors()>0 && n!=get_num_vectors())
		{
			feature_array.pop_back();
			error("Number of feature vectors does not match (expected {}, "
					"obj has {})", get_num_vectors(), n);
		}
		num_vec=n;
	});
}

std::shared_ptr<Features> CombinedFeatures::create_merged_copy(std::shared_ptr<Features> other) const
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

	auto casted = std::dynamic_pointer_cast<CombinedFeatures>(other);

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

	auto result=std::make_shared<CombinedFeatures>();
	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		auto current_this=get_feature_obj(f_idx);
		auto current_other=casted->get_feature_obj(f_idx);

		result->append_feature_obj(
				current_this->create_merged_copy(current_other));
	}

	SG_TRACE("leaving {}::create_merged_copy()", get_name());
	return result;
}

void CombinedFeatures::add_subset(SGVector<index_t> subset)
{
	SG_TRACE("entering {}::add_subset()", get_name());
	std::unordered_set<std::shared_ptr<Features>> processed;

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		auto current=get_feature_obj(f_idx);

		if (processed.find(current) == processed.end())
		{
			/* remember that subset was added here */
			current->add_subset(subset);
			processed.insert(current);
			SG_DEBUG("adding subset to {} at {}",
					current->get_name(), fmt::ptr(current.get()));
		}

	}

	/* also add subset to local stack to have it for easy access */
	m_subset_stack->add_subset(subset);

	subset_changed_post();
	SG_TRACE("leaving {}::add_subset()", get_name());
}

void CombinedFeatures::remove_subset()
{
	SG_TRACE("entering {}::remove_subset()", get_name());
	std::unordered_set<std::shared_ptr<Features>> processed;

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		auto current=get_feature_obj(f_idx);
		if (processed.find(current) == processed.end())
		{
			/* remember that subset was added here */
			current->remove_subset();
			processed.insert(current);
			SG_DEBUG("removing subset from {} at {}",
					current->get_name(), fmt::ptr(current.get()));
		}

	}

	/* also remove subset from local stack to have it for easy access */
	m_subset_stack->remove_subset();

	subset_changed_post();
	SG_TRACE("leaving {}::remove_subset()", get_name());
}

void CombinedFeatures::remove_all_subsets()
{
	SG_TRACE("entering {}::remove_all_subsets()", get_name());
	std::unordered_set<std::shared_ptr<Features>> processed;

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		auto current=get_feature_obj(f_idx);
		if (processed.find(current) == processed.end())
		{
			/* remember that subset was added here */
			current->remove_all_subsets();
			processed.insert(current);
			SG_DEBUG("removing all subsets from {} at {}",
					current->get_name(), fmt::ptr(current.get()));
		}

	}

	/* also remove subsets from local stack to have it for easy access */
	m_subset_stack->remove_all_subsets();

	subset_changed_post();
	SG_TRACE("leaving {}::remove_all_subsets()", get_name());
}

std::shared_ptr<Features> CombinedFeatures::copy_subset(SGVector<index_t> indices) const
{
	/* this is returned with the results of copy_subset of sub-features */
	auto result=std::make_shared<CombinedFeatures>();

	/* map to only copy same feature objects once */
	std::unordered_map<std::shared_ptr<Features>, std::shared_ptr<Features>> processed;
	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		auto current=get_feature_obj(f_idx);

		std::shared_ptr<Features> new_element=NULL;

		/* only copy if not done yet, otherwise, use old copy */
		if (processed.find(current) == processed.end())
		{
			new_element=current->copy_subset(indices);
			processed.emplace(current, new_element);
		}
		else
		{
			new_element=processed[current];
		}

		/* add to result */
		result->append_feature_obj(new_element);
	}
	return result;
}
