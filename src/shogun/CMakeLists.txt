include_directories(. ..)

aux_source_directory(base BASE)
aux_source_directory(classifier CLASSIFIER)
aux_source_directory(classifier/mkl CLASSIFIER_MKL)
aux_source_directory(classifier/svm CLASSIFIER_SVM)
aux_source_directory(classifier/vw CLASSIFIER_WV)
aux_source_directory(classifier/vw/cache CLASSIFIER_WV_CACHE)
aux_source_directory(classifier/vw/learners CLASSIFIER_WV_LEARNERS)
aux_source_directory(clustering CLUSTERING)
aux_source_directory(converter CONVERTER)
aux_source_directory(distance DISTANCE)
aux_source_directory(distributions DISTRIBUTIONS)
aux_source_directory(evaluation EVALUATION)
aux_source_directory(features FEATURES)
aux_source_directory(features/streaming FEATURES_STREAMING)
aux_source_directory(io IO)
aux_source_directory(io/streaming IO_STREAMING)
aux_source_directory(kernel KERNEL)
aux_source_directory(kernel/normalizer KERNEL_NORMALIZER)
aux_source_directory(kernel/string KERNEL_STRING)
aux_source_directory(labels LABELS)
aux_source_directory(latent LATENT)
aux_source_directory(lib LIB)
aux_source_directory(lib/eternal LIB_EXTERNAL)
aux_source_directory(loss LOSS)
aux_source_directory(machine MACHINE)
aux_source_directory(mathematics MATHEMATICS)
aux_source_directory(modelselection MODELSELECTION)
aux_source_directory(multiclass MULTICLASS)
aux_source_directory(multiclass/ecoc MULTICLASS_ECOC)
aux_source_directory(multiclass/tree MULTICLASS_TREE)
aux_source_directory(optimization/lbfgs OPTIMIZATION_LBFGS)
aux_source_directory(optimization/liblinear OPTIMIZATION_LIBLINEAR)
aux_source_directory(preprocessor PREPROCESSOR)
aux_source_directory(regression REGRESSION)
aux_source_directory(regression/svr REGRESSION_SVR)
aux_source_directory(regression/gp REGRESSION_GP)
aux_source_directory(statistics STATISTICS)
aux_source_directory(structure STRUCTURE)
aux_source_directory(transfer/multitask TRANSFER_MULTITASK)
aux_source_directory(transfer/domain_adaptation TRANSFER_DOMAIN_ADAPTATION)
aux_source_directory(ui UI)

add_library(shogun SHARED
                   ${BASE}
                   ${CLASSIFIER}
                     ${CLASSIFIER_MKL}
                     ${CLASSIFIER_SVM}
                     ${CLASSIFIER_WV}
                       ${CLASSIFIER_WV_CACHE}
                       ${CLASSIFIER_WV_LEARNERS}
                   ${CLUSTERING}
                   ${CONVERTER}
                   ${DISTANCE}
                   ${DISTRIBUTIONS}
                   ${EVALUATION}
                   ${FEATURES}
                     ${FEATURES_STREAMING}
                   ${IO}
                     ${IO_STREAMING}
                   ${KERNEL}
                     ${KERNEL_NORMALIZER}
                     ${KERNEL_STRING}
                   ${LABELS}
                   ${LATENT}
                   ${LIB}
                   ${LOSS}
                   ${MACHINE}
                   ${MATHEMATICS}
                   ${MODELSELECTION}
                   ${MULTICLASS}
                     ${MULTICLASS_ECOC}
                     ${MULTICLASS_TREE}
                   ${PREPROCESSOR}
                   ${REGRESSION}
                     ${REGRESSION_SVR}
                     ${REGRESSION_GP}
                   ${STATISTICS}
                   ${STRUCTURE}
                   ${UI})

