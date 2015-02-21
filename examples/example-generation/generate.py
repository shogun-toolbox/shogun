#!/usr/bin/python

import os
from parse import parse
from translate import translate
import json
import argparse

def translateExamples(inputDir, outputDir, targetsDir, includedTargets=None):
    # Load all target dictionaries
    targets = []
    for target in os.listdir(targetsDir):
        # Ignore targets not in includedTargets
        if includedTargets and not os.path.basename(target).split(".")[0] in includedTargets:
            continue

        with open(os.path.join(targetsDir, target)) as tFile:
            targets.append(json.load(tFile))

    # Translate each example
    for f in os.listdir(inputDir):
        # Parse the example file
        ast = parse(os.path.join(inputDir,f))
        basename = f[:-len(".sg")]

        # Translate ast to each target language
        for target in targets:
            translation = translate(ast, targetDict=target)
            directory = os.path.join(outputDir, target["OutputDirectoryName"])
            extension = target["FileExtension"]
            
            # Create directory if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Write translation
            with open(os.path.join(directory, basename + extension), "w") as nf:
                nf.write(translation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="path to output directory")
    parser.add_argument("-i", "--input", help="path to examples directory (input)")
    parser.add_argument("-t", "--targetsfolder", help="path to directory with target JSON files")
    parser.add_argument('targets', nargs='*', help="Targets to include (one or more of: python java r octave). If not specified all targets are produced.")

    args = parser.parse_args()

    outputDir = "outputs"
    if args.output:
        outputDir = args.output

    inputDir = "examples"
    if args.input:
        inputDir = args.input

    targetsDir = "targets"
    if args.targetsfolder:
        targetsDir = args.targetsfolder

    translateExamples(inputDir, outputDir, targetsDir, args.targets)
