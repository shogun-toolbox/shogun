/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Saurabh Mahindre, Evgeniy Andreev, Wu Lin, Vladislav Horbatiuk,
 *          Yuyu Zhang, Bjoern Esser, Soumyajit De
 */

#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/base/SGObject.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/base/range.h>

namespace shogun
{
	class File;
	class Preprocessor;
	class Kernel;
}

namespace shogun
{

/** @brief The class Features is the base class of all feature objects.
 *
 * It can be understood as a dense real valued feature matrix (with e.g.
 * columns as single feature vectors), a set of strings, graphs or any other
 * arbitrary collection of objects. As a result this class is kept very general
 * and implements only very weak interfaces to
 *
 * - duplicate the Feature object
 * - obtain the feature type (like F_DREAL, F_SHORT ...)
 * - obtain the feature class (like Simple dense matrices, sparse or strings)
 * - obtain the number of feature "vectors"
 *
 *   In addition it provides helpers to check e.g. for compatibility of feature objects.
 *
 *   Currently there are 3 general feature classes, which are DenseFeatures
 *   (dense matrices), SparseFeatures (sparse matrices), CStringFeatures (a
 *   set of strings) from which all the specific features like DenseFeatures<float64_t>
 *   (dense real valued feature matrices) are derived.
 *
 *
 * (Multiple) Subsets (of subsets) are supported.
 * Sub-classes may want to overwrite the subset_changed_post() method which is
 * called automatically after each subset change. See method documentations to
 * see how behaviour is changed when subsets are active.
 * A subset is put onto a stack using the add_subset() method. The last added
 * subset may be removed via remove_subset(). There is also the possibility to
 * add subsets in place (this only stores one index vector in memory as opposed
 * to many when add_subset() is called many times) with add_subset_in_place().
 * The latter does not allow to remove such modifications one-by-one.
 */
class Features : public SGObject
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		Features(int32_t size=0);

		/** copy constructor */
		Features(const Features& orig);

		/** constructor
		 *
		 * @param loader File object via which data shall be loaded
		 */
		Features(std::shared_ptr<File> loader);

		/** duplicate feature object
		 *
		 * abstract base method
		 *
		 * @return feature object
		 */
		virtual std::shared_ptr<Features> duplicate() const=0;

		~Features() override;

		/** get feature type
		 *
		 * abstract base method
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const=0;

		/** get feature class
		 *
		 * abstract base method
		 *
		 * @return feature class like STRING, SIMPLE, SPARSE...
		 */
		virtual EFeatureClass get_feature_class() const=0;

#ifndef SWIG
		/** returns an iterator of indices
		 * from 0 to @ref Features::get_num_vectors
		 *
		 * Should be used in algorithms in the following way:
		 * @code
		 * for (auto idx : features->index_iterator()) { ... }
		 * @endcode
		 *
		 */
		virtual Range<int32_t> index_iterator() const
		{
			return range(0, get_num_vectors());
		}
#endif

		/** add preprocessor
		 *
		 * @param p preprocessor to set
		 */
		virtual void add_preprocessor(std::shared_ptr<Preprocessor> p);

		/** delete preprocessor from list
		 *
		 * @param num index of preprocessor in list
		 */
		virtual void del_preprocessor(int32_t num);

		/** get specified preprocessor
		 *
		 * @param num index of preprocessor in list
		 */
		std::shared_ptr<Preprocessor> get_preprocessor(int32_t num) const;

		/** get number of preprocessors
		 *
		 * @return number of preprocessors
		 */
		int32_t get_num_preprocessors() const;

		/** clears all preprocs */
		void clean_preprocessors();

		/** print preprocessors */
		void list_preprocessors();

		/** get cache size
		 *
		 * @return cache size
		 */
		int32_t get_cache_size() const;

		/** get number of examples/vectors, possibly corresponding to the current subset
		 *
		 * abstract base method
		 *
		 * @return number of examples/vectors (possibly of subset, if implemented)
		 */
		virtual int32_t get_num_vectors() const=0;

		/** in case there is a feature matrix allow for reshaping
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num_features new number of features
		 * @param num_vectors new number of vectors
		 * @return if reshaping was successful
		 */
		virtual bool reshape(int32_t num_features, int32_t num_vectors);

		/** load features from file
		 *
		 * @param loader File object via which data shall be loaded
		 */
		virtual void load(std::shared_ptr<File> loader);

		/** save features to file
		 *
		 * @param writer File object via which data shall be saved
		 */
		virtual void save(std::shared_ptr<File> writer);

		/** check feature compatibility
		 *
		 * @param f features to check for compatibility
		 * @return if features are compatible
		 */
		bool check_feature_compatibility(const std::shared_ptr<Features>& f) const;

		/** check if features have given property
		 *
		 * @param p feature property
		 * @return if features have given property
		 */
		bool has_property(EFeatureProperty p) const;

		/** set property
		 *
		 * @param p kernel property to set
		 */
		void set_property(EFeatureProperty p);

		/** unset property
		 *
		 * @param p kernel property to unset
		 */
		void unset_property(EFeatureProperty p);

		/** Takes a list of feature instances and returns a new instance being
		 * a concatenation of a copy of this instace's data and the given
		 * instancess data. Note that the feature types have to be equal.
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param others list of feature objects to append
		 * @return new feature object which contains copy of data of this
		 * instance and given ones
		 */
		virtual std::shared_ptr<Features> create_merged_copy(const std::vector<std::shared_ptr<Features>>& others) const
		{
			error("{}::create_merged_copy() is not yet implemented!");
			return NULL;
		}

		/** Convenience method for method with same name and list as parameter.
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param other feature object to append
		 * @return new feature object which contains copy of data of this
		 * instance and of given one
		 */
		virtual std::shared_ptr<Features> create_merged_copy(std::shared_ptr<Features> other) const
		{
			error("{}::create_merged_copy() is not yet implemented!");
			return NULL;
		}

		/** Adds a subset of indices on top of the current subsets (possibly
		 * subset of subset). Every call causes a new active index vector
		 * to be stored. Added subsets can be removed one-by-one. If this is not
		 * needed, add_subset_in_place() should be used (does not store
		 * intermediate index vectors)
		 *
		 * Calls subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_subset(SGVector<index_t> subset);

		/** Sets/changes latest added subset. This allows to add multiple subsets
		 * with in-place memory requirements. They cannot be removed one-by-one
		 * afterwards, only the latest active can. If this is needed, use
		 * add_subset(). If no subset is active, this just adds.
		 *
		 * Calls subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to replace the latest one with.
		 * */
		virtual void add_subset_in_place(SGVector<index_t> subset);

		/** removes that last added subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_subset();

		/** removes all subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_subsets();

		/** returns subset stack
		 *
		 * @return subset stack
		 */
		virtual std::shared_ptr<SubsetStack> get_subset_stack() const;

		/** method may be overwritten to update things that depend on subset */
		virtual void subset_changed_post() {}

		/** Creates a new Features instance containing copies of the elements
		 * which are specified by the provided indices.
		 *
		 * This method is needed for a KernelMachine to store its model data.
		 * NOT IMPLEMENTED!
		 *
		 * @param indices indices of feature elements to copy
		 * @return new Features instance with copies of feature data
		 */
		virtual std::shared_ptr<Features> copy_subset(SGVector<index_t> indices) const;

		/** Creates a new Features instance containing only the dimensions
		 * of the feature vector which are specified by the provided indices.
		 *
		 * This method is needed for feature selection tasks
		 * NOT IMPLEMENTED!
		 *
		 * @param dims indices of feature dimensions to copy
		 * @return new Features instance with copies of specified features
		 */
		virtual std::shared_ptr<Features> copy_dimension_subset(SGVector<index_t> dims) const;

		/** does this class support compatible computation bewteen difference classes?
		 * for example, this->dot(rhs_prt),
		 * can rhs_prt be an instance of a difference class?
		 *
		 * @return whether this class supports compatible computation
		 */
		virtual bool support_compatible_class() const {return false;}

		/** Given a class in right hand side, does this class support compatible computation?
		 *
		 * for example, is this->dot(rhs_prt) valid,
		 * where rhs_prt is the class in right hand side
		 *
		 * @param rhs the class in right hand side
		 * @return whether this class supports compatible computation
		 */
		virtual bool get_feature_class_compatibility (EFeatureClass rhs) const;

#ifndef SWIG // SWIG should skip this part
		virtual std::shared_ptr<Features> shallow_subset_copy()
		{
			not_implemented(SOURCE_LOCATION);;
			return NULL;
		}
#endif

	private:
		void init();

	private:
		/** feature properties */
		uint64_t  properties;

		/** size of cache in MB */
		int32_t cache_size;

		/** list of preprocessors */
		std::vector<std::shared_ptr<Preprocessor>> preproc;

	protected:
		/** subset used for index transformations */
		std::shared_ptr<SubsetStack> m_subset_stack;
};
}
#endif
