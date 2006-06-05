

	bool normalize_data();
	/// compute the indices (i.e. the non-constant dimension index vector) from svm_file
	bool compute_index(CObservatio* svm_file);
	/// compute the cov^{-1/2} matrix from svm_file
	bool compute_cov(FILE* svm_file);
	/// compute the mean-vector from svm_file
	bool compute_mean(FILE* svm_file);

	/// load the indices (i.e. the non-constant dimension index vector) from svm_file
	bool load_index(FILE* svm_file);
	/// load the cov^{-1/2} matrix from svm_file
	bool load_cov(FILE* svm_file);
	/// load the mean-vector from svm_file
	bool load_mean(FILE* svm_file);

	/// appends the indices (i.e. the non-constant dimension index vector) to svm_file
	bool save_index(FILE* svm_file);
	/// appends the cov^{-1/2} matrix to svm_file
	bool save_cov(FILE* svm_file);
	/// appends the mean-vector to svm_file
	bool save_mean(FILE* svm_file);
