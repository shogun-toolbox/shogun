from modshogun import StreamingVwFile
from modshogun import T_SVMLIGHT
from modshogun import StreamingVwFeatures
from modshogun import VowpalWabbit

def run_vw():
    """Runs the VW algorithm on a toy dataset in SVMLight format."""

    # Open the input file as a StreamingVwFile
    input_file = StreamingVwFile("../data/fm_train_sparsereal.dat")

    # Tell VW that the file is in SVMLight format
    # Supported types are T_DENSE, T_SVMLIGHT and T_VW
    input_file.set_parser_type(T_SVMLIGHT)

    # Create a StreamingVwFeatures object, `True' indicating the examples are labelled
    features = StreamingVwFeatures(input_file, True, 1024)

    # Create a VW object from the features
    vw = VowpalWabbit(features)

    # Train
    vw.train()

if __name__ == "__main__":
    run_vw()
