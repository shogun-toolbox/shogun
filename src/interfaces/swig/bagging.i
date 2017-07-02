/* remove C prefix */
%rename(BaggingMachine) CBaggingMachine;
%rename(RandomForest) CRandomForest;

/* include class headers to make them visible from target language */
%include <shogun/machine/BaggingMachine.h>
%include <shogun/machine/RandomForest.h>
