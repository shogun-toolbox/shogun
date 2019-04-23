/* remove C prefix */
%shared_ptr(shogun::BaggingMachine)
%shared_ptr(shogun::RandomForest)

/* include class headers to make them visible from target language */
%include <shogun/machine/BaggingMachine.h>
%include <shogun/machine/RandomForest.h>
