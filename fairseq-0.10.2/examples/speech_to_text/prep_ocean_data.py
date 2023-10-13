#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import os.path as op
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from data_utils import (
    create_zip,
    extract_fbank_features,
    extract_fbank_features_wav,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

from scipy.io import wavfile
import soundfile as sf

log = logging.getLogger(__name__)

SPLITS = [
    "train",
    "test",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    os.makedirs(args.output_root, exist_ok=True)
    # Extract features
    feature_root = op.join(args.output_root, "fbank80")
    os.makedirs(feature_root, exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        datasetFile = op.join(args.output_root,split,'dataset')
        dataset = open(datasetFile,'r').readlines()
        print("Extracting log mel filter bank features...")
        for line in tqdm(dataset):
            spk_id,utt_id,text,wavFile = line.strip().split('\t')

            # .wav file
            # wavFile=op.join(args.output_root,wavFile)
            # sample_rate,wav = wavfile.read(wavFile)

            # .flac file 
            wavFile=op.join(args.output_root,wavFile[:-3]+'FLAC')
            wav,sample_rate = sf.read(wavFile)

            sample_id = f"{spk_id}-{utt_id}"
            extract_fbank_features_wav(
                wav, sample_rate, op.join(feature_root, f"{sample_id}.npy")
            )
    # Pack features into ZIP
    zip_filename = "fbank80.zip"
    zip_path = op.join(args.output_root, zip_filename)
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(args.output_root, zip_filename)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}

        datasetFile = op.join(args.output_root,split,'dataset')
        dataset = open(datasetFile,'r').readlines()

        for line in tqdm(dataset):
            spk_id,utt_id,text,wavFile = line.strip().split('\t')
            # .wav file
            # wavFile=op.join(args.output_root,wavFile)
            # sample_rate,wav = wavfile.read(wavFile)

            # .flac file 
            wavFile=op.join(args.output_root,wavFile[:-3]+'FLAC')
            wav,sample_rate = sf.read(wavFile)

            sample_id = f"{spk_id}-{utt_id}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(zip_manifest[sample_id])
            duration_ms = int(wav.size / sample_rate * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["tgt_text"].append(text)
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), op.join(args.output_root, f"{split}.tsv")
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])
    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            f.name,
            op.join(args.output_root, spm_filename_prefix),
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        args.output_root, spm_filename_prefix + ".model", specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
