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
aux_source_directory(io IO)
aux_source_directory(kernel KERNEL)
aux_source_directory(lib LIB)
aux_source_directory(loss LOSS)
aux_source_directory(machine MACHINE)
aux_source_directory(mathematics MATHEMATICS)
aux_source_directory(modelselection MODELSELECTION)
aux_source_directory(preprocessor PREPROCESSOR)
aux_source_directory(regression REGRESSION)
aux_source_directory(regression/svr REGRESSION_SVR)
aux_source_directory(structure STRUCTURE)
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
                   ${IO}
                   ${KERNEL}
                   ${LIB}
                   ${LOSS}
                   ${MACHINE}
                   ${MATHEMATICS}
                   ${MODELSELECTION}
                   ${PREPROCESSOR}
                   ${REGRESSION}
                     ${REGRESSION_SVR}
                   ${STRUCTURE}
                   ${UI})

