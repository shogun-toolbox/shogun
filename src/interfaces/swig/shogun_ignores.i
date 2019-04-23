%ignore *::operator[];
%ignore *::operator=;
%ignore *::operator();
#if !defined(SWIGPERL)
%ignore shogun::SGVector::operator+=;
%ignore shogun::SGVector::operator+;
%ignore *::operator==;
%ignore *::operator!=;
#endif
#if defined(SWIGPERL)
%ignore shogun::SGObject::next;
%ignore shogun::bmrm_ll::next;
#endif

%ignore *::operator bool*;
%ignore *::operator char*;
%ignore *::operator unsigned char*;
%ignore *::operator unsigned short*;
%ignore *::operator int*;
%ignore *::operator long*;
%ignore *::operator unsigned long*;
%ignore *::operator float*;
%ignore *::operator double*;
%ignore *::operator std::complex<double>*;

#ifdef SWIGJAVA
%ignore SGIO;
#endif

%ignore ref;
%ignore unref;

%ignore shogun::SGVector::display_vector;
%ignore shogun::SGMatrix::display_matrix;

%ignore shogun::ICP_stats;
%ignore shogun::bmrm_ll;
%ignore shogun::TMultipleCPinfo;
%ignore refcount_t;
%ignore QP;
%ignore ConsensusEntry;
%ignore DNATrie;
%ignore Model;
%ignore SparsityStructure;

#if !defined(SWIGPERL)
/*%rename("%s") *::LatentModel; */
/* ../../shogun/lib/SGReferencedData.h:66: Warning 362: operator= ignored */
%ignore shogun::SGReferencedData::operator=;

/* ../../shogun/lib/DynamicArray.h:589: Warning 362: operator= ignored */
/* ../../shogun/base/DynArray.h:502: Warning 362: operator= ignored */
/* ../../shogun/lib/Trie.h:175: Warning 362: operator= ignored */
/* ../../shogun/lib/DynamicObjectArray.h:418: Warning 362: operator= ignored */
/* ../../shogun/classifier/mkl/MKLMulticlass.h:105: Warning 362: operator=  */
#endif

%ignore substring;
%ignore LaRankOutput;
%ignore larank_kcache_s;
%ignore LaRankPattern;
%ignore LaRankPatterns;
%ignore POIMTrie;
%ignore segment_loss_struct;
%ignore joint_list_struct;
%ignore TreeParseInfo;
%ignore T_ATTRIBUTE;
%ignore T_ALPHA_BETA;
%ignore TSGDataType;
%ignore T_HMM_INDIZES;
%ignore SSKTripleFeature;
%ignore SSKFeatures;
%ignore SSKDoubleFeature;
%ignore quadratic_program;
%ignore __STDC_FORMAT_MACROS;
%ignore shogun::SGSparseVector::SGSparseVector(SGSparseVectorEntry, index_t, index_t, bool);
%ignore shogun::SGSparseVector::features;
%ignore shogun::SGSparseVectorEntry;
%ignore shogun::CParameter;

%ignore shogun::SGString;
%ignore sparse_dot;
%ignore dense_dot;
%ignore add_to_dense_vec;
%ignore dense_dot_range;
%ignore dense_dot_range_subset;
%ignore dense_dot_range_helper;
%ignore get_feature_iterator;
%ignore get_next_feature;
%ignore free_feature_iterator;
%ignore compute_sparse_feature_vector;

%ignore shogun::KNN::m_covertree;
%ignore shogun::KNN_COVERTREE_POINT;
%ignore free_feature_vector;
%ignore free_sparse_feature_vector;
%ignore shogun::CNode;
%ignore shogun::CTaxonomy::get_node;
%ignore shogun::CTaxonomy::add_node;
%ignore shogun::CTaxonomy::intersect_root_path;
%ignore shogun::SGVector<shogun::Gaussian*>;
%ignore shogun::CGMM::CGMM(const SGVector<shogun::Gaussian*>&, const shogun::SGVector<float64_t>&, bool);
%ignore shogun::CGMM::get_comp;
%ignore shogun::CGMM::set_comp;
%ignore shogun::DenseFeatures::dense_feature_iterator;;
%ignore shogun::DenseFeatures::DenseFeatures(ST*, int32_t, int32_t);
%ignore shogun::DenseFeatures::get_feature_vector(int32_t, int32_t&, bool&);
%ignore shogun::DenseFeatures::set_feature_matrix(ST*, int32_t, int32_t);
%ignore shogun::DenseFeatures::vector_subset;
%ignore shogun::DenseFeatures::feature_subset;
%ignore shogun::DenseFeatures::get_feature_matrix(ST**, int32_t*, int32_t*);
%ignore shogun::DenseFeatures::get_feature_matrix(int32_t&, int32_t&);
%ignore shogun::DenseFeatures::get_transposed(int32_t&, int32_t&);
%ignore shogun::DenseFeatures::dense_dot(int32_t, const float64_t*, int32_t);
%ignore shogun::DenseFeatures::add_to_dense_vec(float64_t, int32_t, float64_t*, int32_t, bool );
%ignore shogun::SparseFeatures::sparse_feature_iterator;;
%ignore shogun::SparseFeatures::SparseFeatures(shogun::SGSparseVector<shogun::ST>*, int32_t, int32_t, bool);
%ignore shogun::SparseFeatures::get_full_feature_vector(int32_t, int32_t&);
%ignore shogun::SparseFeatures::get_sparse_feature_matrix(int32_t&, int32_t&);
%ignore shogun::SparseFeatures::compute_squared;
%ignore shogun::SparseFeatures::compute_squared_norm;
%ignore shogun::SparseFeatures::get_transposed(int32_t&, int32_t&);
%ignore shogun::SparseFeatures::clean_tsparse;

%ignore shogun::CStringFeatures::get_feature_vector(int32_t, int32_t&, bool&);
%ignore shogun::CStringFeatures::set_features(SGString<ST>*, int32_t, int32_t);
%ignore shogun::CStringFeatures::append_features(SGString<ST>*, int32_t, int32_t);
%ignore shogun::CStringFeatures::get_features(int32_t&, int32_t&);
%ignore shogun::CStringFeatures::get_transposed(int32_t&, int32_t&);
%ignore shogun::CStringFeatures::get_features(SGString<ST>**, int32_t*);
%ignore shogun::CStringFeatures::copy_features(int32_t&, int32_t&);
%ignore shogun::CStringFeatures::get_zero_terminated_string_copy(SGString<ST>);
%ignore shogun::CStringFeatures::unembed_word;
%ignore shogun::CStringFeatures::embed_word;
%ignore shogun::CStringFeatures::set_feature_vector(int32_t, ST*, int32_t);
%ignore shogun::CStringFeatures::get_histogram;
%ignore shogun::CStringFeatures::create_random(float64_t*, int32_t, int32_t, int32_t);

%ignore shogun::SVMLight::add_to_index;
%ignore shogun::SVMLight::calculate_svm_model;
%ignore shogun::SVMLight::call_mkl_callback;
%ignore shogun::SVMLight::check_optimality;
%ignore shogun::SVMLight::clear_index;
%ignore shogun::SVMLight::compute_index;
%ignore shogun::SVMLight::compute_matrices_for_optimization;
%ignore shogun::SVMLight::compute_matrices_for_optimization_parallel;
%ignore shogun::SVMLight::compute_objective_function;
%ignore shogun::SVMLight::get_runtime;
%ignore shogun::SVMLight::init;
%ignore shogun::SVMLight::init_shrink_state;
%ignore shogun::SVMLight::optimize_svm;
%ignore shogun::SVMLight::optimize_to_convergence;
%ignore shogun::SVMLight::reactivate_inactive_examples;
%ignore shogun::SVMLight::select_next_qp_subproblem_grad;
%ignore shogun::SVMLight::select_next_qp_subproblem_rand;
%ignore shogun::SVMLight::select_top_n;
%ignore shogun::SVMLight::shrink_problem;
%ignore shogun::SVMLight::shrink_state_cleanup;
%ignore shogun::SVMLight::svm_learn;
%ignore shogun::SVMLight::update_linear_component;
%ignore shogun::SVMLight::update_linear_component_mkl;
%ignore shogun::SVMLight::update_linear_component_mkl_linadd;
%ignore shogun::SVMLight::update_linear_component_mkl_linadd_helper;
%ignore shogun::SVRLight::call_mkl_callback;
%ignore shogun::SVRLight::svr_learn;
%ignore shogun::SVRLight::update_linear_component_mkl;
%ignore shogun::SVRLight::update_linear_component_mkl_linadd;

%ignore shogun::CStringWordFeatures::cleanup;
%ignore shogun::CStringWordFeatures::cleanup_feature_vector;
%ignore shogun::CStringWordFeatures::cleanup_feature_vectors;

%ignore shogun::CSGDQN_combine_and_clip;
%ignore shogun::CSGDQN_compute_ratio;

%ignore shogun::SerializableFile::read_cont_begin;
%ignore shogun::SerializableFile::read_cont_end;
%ignore shogun::SerializableFile::read_item_begin;
%ignore shogun::SerializableFile::read_item_end;
%ignore shogun::SerializableFile::read_scalar;
%ignore shogun::SerializableFile::read_sgserializable_begin;
%ignore shogun::SerializableFile::read_sgserializable_end;
%ignore shogun::SerializableFile::read_sparse_begin;
%ignore shogun::SerializableFile::read_sparse_end;
%ignore shogun::SerializableFile::read_sparseentry_begin;
%ignore shogun::SerializableFile::read_sparseentry_end;
%ignore shogun::SerializableFile::read_string_begin;
%ignore shogun::SerializableFile::read_string_end;
%ignore shogun::SerializableFile::read_stringentry_begin;
%ignore shogun::SerializableFile::read_stringentry_end;
%ignore shogun::SerializableFile::read_type_begin;
%ignore shogun::SerializableFile::read_type_end;
%ignore shogun::SerializableFile::write_cont_begin;
%ignore shogun::SerializableFile::write_cont_end;
%ignore shogun::SerializableFile::write_item_begin;
%ignore shogun::SerializableFile::write_item_end;
%ignore shogun::SerializableFile::write_scalar;
%ignore shogun::SerializableFile::write_sgserializable_begin;
%ignore shogun::SerializableFile::write_sgserializable_end;
%ignore shogun::SerializableFile::write_sparse_begin;
%ignore shogun::SerializableFile::write_sparse_end;
%ignore shogun::SerializableFile::write_sparseentry_begin;
%ignore shogun::SerializableFile::write_sparseentry_end;
%ignore shogun::SerializableFile::write_string_begin;
%ignore shogun::SerializableFile::write_string_end;
%ignore shogun::SerializableFile::write_stringentry_begin;
%ignore shogun::SerializableFile::write_stringentry_end;
%ignore shogun::SerializableFile::write_type_begin;
%ignore shogun::SerializableFile::write_type_end;
>>>>>>> drop C prefix

%ignore shogun::CMosek;

%ignore shogun::FactorType::FactorType();
%ignore shogun::TableFactorType::TableFactorType();
%ignore shogun::FactorDataSource::FactorDataSource();
%ignore shogun::Factor::Factor();
%ignore shogun::Factor::Factor(TableFactorType*, SGVector<int32_t>, SGSparseVector<float64_t>);
%ignore shogun::Factor::Factor(TableFactorType*, SGVector<int32_t>, FactorDataSource*);
%ignore shogun::CDisjointSet::CDisjointSet();
%ignore shogun::FactorGraph::FactorGraph();
%ignore shogun::MAPInference::MAPInference();
%ignore shogun::CGraphCut::CGraphCut();
%ignore shogun::FactorGraphModel::FactorGraphModel();

%ignore shogun::Range;
