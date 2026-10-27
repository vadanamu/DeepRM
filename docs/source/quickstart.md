# 🚀 Quickstart
* The repository has example data in `example/`. Run the commands below from the repository folder.
* The inference example (`example/call-example`) holds 10,225 HEK293T mRNA reads, basecalled with `rna004_sup@v6.0.0`
  and aligned to transcripts.
* To run your own POD5 files, see {doc}`usage`.

## RNA Modification Detection
* Estimated time: about 3 minutes on one B200 GPU.

1️⃣ **Prepare data**
```bash
deeprm call prep -p example/call-example/call_example_pod5 -b example/call-example/call_example.bam -o <prep_dir>
```

2️⃣ **Run inference**
```bash
deeprm call run -b example/call-example/call_example.bam -i <prep_dir> -o <pred_dir>
```
* The site-level result is `<pred_dir>/<prep_dir name>/site-level/pileup.bed`. All output files are listed in
  {ref}`output-files`.

## Model Training
* Estimated time: about 1 hour.

1️⃣ **Prepare unmodified & modified training data**
```bash
deeprm train prep -p example/train-example/A/train_example_A_pod5 -b example/train-example/A/train_example_A.bam -o <prep_dir>/a
deeprm train prep -p example/train-example/m6A/train_example_m6A_pod5 -b example/train-example/m6A/train_example_m6A.bam -o <prep_dir>/m6a
```

2️⃣ **Compile training data**
```bash
deeprm train compile -n <prep_dir>/a/data -p <prep_dir>/m6a/data -o <prep_dir>/compiled -c 1000
```
* The example is small, so `-c 1000` writes smaller files than the default (4000). With the default, the
  validation set would be empty and the command stops.

3️⃣ **Run training**
```bash
deeprm train run --data <prep_dir>/compiled --output <output_dir> --batch 64
```
* The run writes the trained model file (.pt). Training options: {doc}`usage`.
