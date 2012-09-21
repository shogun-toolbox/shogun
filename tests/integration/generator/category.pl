CLASSIFIER=0
CLUSTERING=1
DISTANCE=2
DISTRIBUTION=3
KERNEL=4
PREPROC=5
REGRESSION=6


def get_name(catID):
	"""
	Get index to lookup the entity's name in the output data dictionary.

	@param catID ID of the category
	@return string index for name lookup in output data
	"""

	names={
		CLASSIFIER: 'classifier_name',
		CLUSTERING: 'clustering_name',
		DISTANCE: 'distance_name',
		DISTRIBUTION: 'distribution_name',
		KERNEL: 'kernel_name',
		PREPROC: 'preproc_name',
		REGRESSION: 'regression_name',
	}

	try:
		return names[catID]
	except KeyError:
		return ''


def get_as_string (catID):
	"""Returns the string representation of given category.

	@param catID ID of a category
	@return string of the category or empty string if ID was invalid
	"""

	strings={
		CLASSIFIER: 'classifier',
		CLUSTERING: 'clustering',
		DISTANCE: 'distance',
		DISTRIBUTION: 'distribution',
		KERNEL: 'kernel',
		PREPROC: 'preproc',
		REGRESSION: 'regression'
	}

	try:
		return strings[catID]
	except KeyError:
		return ''
