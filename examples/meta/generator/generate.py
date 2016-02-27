#!/usr/bin/python

import os
from parse import parse
from translate import translate
import json
import argparse

def subfilesRelative(directory, filter_by):
    """ Returns a generator of pairs (subpath, filename) in a given directory.
        Filenames are filtered by the filter_by function.
    """
    for dir_, _, files in os.walk(directory):
        for file in files:
            if filter_by(file):
                yield os.path.relpath(dir_, inputDir), file

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
    for dirRelative, filename in subfilesRelative(inputDir, filter_by=lambda x: x.lower().endswith('.sg')):
        # Parse the example file
        with open(os.path.join(inputDir, dirRelative, filename), 'r') as file:
            ast = parse(file.read(), os.path.join(dirRelative, filename))
        basename = os.path.splitext(filename)[0]

        # Translate ast to each target language
        for target in targets:
            translation = translate(ast, targetDict=target)
            directory = os.path.join(outputDir, target["OutputDirectoryName"])
            extension = target["FileExtension"]

            # Create directory if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Write translation
            outputFile = os.path.join(directory, dirRelative, os.path.splitext(filename)[0]+extension)

            # create subdirectories if they don't exist yet
            try:
                os.makedirs(os.path.dirname(outputFile))
            except OSError:
                pass

            with open(outputFile, "w") as nf:
                nf.write(translation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="path to output directory")
    parser.add_argument("-i", "--input", help="path to examples directory (input)")
    parser.add_argument("-t", "--targetsfolder", help="path to directory with target JSON files")
    parser.add_argument('targets', nargs='*', help="Targets to include (one or more of: python java r octave csharp ruby cpp lua). If not specified all targets are produced.")

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
