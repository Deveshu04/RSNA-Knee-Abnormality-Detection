# RSNA Knee Abnormality Detection

Twelve knee MRI findings predicted from the images alone, for the Kaggle competition [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). Training labels are read from the radiology reports by Qwen3-8B and averaged with three public label tables. Three CNN families read a bag of 3-slice windows from five series per study, a small transformer pools the windows into one prediction per finding, and the final submission rank-averages 15 fold models. Every number below comes from executed Kaggle notebooks, their logs, or Kaggle's scoring of the submissions.

## The competition

The Radiological Society of North America (RSNA) asked for a model that reads a knee MRI study and gives the probability of twelve findings:

| Group | Findings |
|---|---|
| Ligaments | anterior cruciate ligament (ACL) tear, medial collateral ligament (MCL) tear |
| Menisci | medial meniscal tear, lateral meniscal tear |
| Cartilage and bone | osteoarthritis of the medial, lateral and patellofemoral compartments, bone contusion, fracture |
| Joint | effusion, synovitis, Baker's (popliteal) cyst |

Predictions are scored by the mean ROC AUC over the twelve findings. It is a code competition: a notebook submitted to Kaggle runs on about 1,300 hidden test studies within 9 hours, offline, and the test studies come without their reports, so the model has to work from the images.

## The data

Notebook 01 describes the data in detail; in short:

- 4,407 training studies with 24,371 DICOM series, 3 to 14 per study, in sagittal, coronal and axial planes, each flagged as fluid-sensitive or not.
- Every training study has its free-text radiology report, in about nine languages (39% English, then Spanish, Turkish, Croatian, Greek, German, a Cyrillic-script language, Dutch and French).
- Only 58 training studies carry radiologist labels for the twelve findings. Every other study has to be labelled from its report, and the 58 serve as a small gold-standard check.
- The scans come from five scanner vendors, with a field of view of 160 mm at the median and slices of about 3 mm. The DICOM laterality tag, which says whether it is a left or a right knee, is present in only half of the studies.

This project uses only the competition data, public label tables derived from its reports, and public pretrained weights.

## Results

### Test scores

Kaggle scores a submission on the hidden test set; the public part of that score is shown during the competition and the rest when it closes on 22 October 2026. The two submissions chosen as final entries:

| Submission (Kaggle kernel) | Models | Gold-58 AUC | Public test AUC |
|---|---|---|---|
| rsna-knee-h3 | RadImageNet ResNet50 with spatial pooling + the same with average pooling + ImageNet SE-ResNeXt50, 5 folds each, equal-weight rank average | 0.919 | 0.940 |
| rsna-knee-h5 | RadImageNet ResNet50 with spatial pooling (weight 2) + with average pooling (weight 1), 5 folds each | 0.918 | 0.939 |

How the score moved over 44 submissions (selected steps; each was one submission):

| Date | Change | Public test AUC |
|---|---|---|
| 30 Sep | ResNet34, 3 views x 24 slices at 160 px, Qwen3-8B report labels | 0.833 |
| 3 Oct | Own pipeline: ResNet50 bag of 3-slice windows, labels averaged from four sources, 5 folds | 0.859 |
| 4 Oct | 224 px / 140 mm crops instead of 192 px / 150 mm, batch 8, 20 epochs; SE-ResNeXt50 fold 0 added | 0.913 |
| 4 Oct | SE-ResNeXt50 on all 5 folds | 0.918 |
| 4 Oct | RadImageNet ResNet50 5 folds instead of ImageNet ResNet50 | 0.925 |
| 5 Oct | RadImageNet with centre-slice first-layer initialisation | 0.929 |
| 6 Oct | Spatial pooling and no-flip variants trained inside the scoring run | 0.931 |
| 10 Oct | The three models trained in the scoring run use raw labels instead of distilled labels | 0.935 |
| 13 Oct | RadImageNet with spatial pooling on raw labels, 4 folds, alone | 0.938 |
| 14 Oct | Three raw-label families, 5 folds each (rsna-knee-h3) | 0.940 |

The last day's five submissions, all on raw labels with 5 folds per family:

| Submission | Families | Gold-58 AUC | Public test AUC |
|---|---|---|---|
| H1 | spatial pooling alone | 0.920 | 0.938 |
| H2 | spatial pooling + average pooling | 0.917 | 0.930 |
| H3 | spatial pooling + average pooling + SE-ResNeXt50 | 0.919 | 0.940 |
| H4 | spatial pooling + average pooling, plus two models trained on all studies inside the scoring run | not measurable, trains on gold | 0.939 |
| H5 | spatial pooling (weight 2) + average pooling | 0.918 | 0.939 |

What the numbers say:

- Labels decide most of the score. With only 58 radiologist-labelled studies, the training targets are read from free-text reports. A plain mean of four report readers (0.897 against the 58 gold studies) beats each of them alone (0.835 to 0.893).
- The 58 gold studies are a noisy ruler. Distilling the labels with the models' own out-of-fold predictions raised label quality on gold from 0.897 to 0.922, yet models trained on the distilled labels scored 0.931 on the test set against 0.935 on the raw labels. From then on every model was trained on raw labels, and decisions were made on the test score.
- RadImageNet weights need a greyscale input. They were pretrained on single-channel images copied to RGB, so three adjacent slices fed as RGB are out of distribution. Initialising the first layer so that the centre slice carries the summed RGB weights and the side slices start at zero raised fold 0 validation from 0.877 to 0.887 and gold from 0.887 to 0.895; on the test set, replacing the old RadImageNet models with centre-initialised ones moved the score from 0.925 to 0.929.
- Gains at the end were small. The spatial-pooling family alone scored 0.938, and the best ensemble 0.940. On gold all five final candidates sit between 0.917 and 0.920, inside the noise of 58 studies.
- The public test score was steady to about 0.001 for most of the work: a rerun with a new seed reproduced 0.931, and family weights of 0.5, 1 and 2 gave 0.929, 0.931 and 0.930. H2 is the exception: it differs from H5 only in one family weight, yet scored 0.930 against 0.939. Its log is normal and the cause is unknown; one candidate is the laterality model, which is retrained in every scoring run and can call a few test knees differently.

### Report labels

Each source labels all 4,407 training studies with a probability per finding. Scores are the mean ROC AUC over the twelve findings on the 58 studies with radiologist labels.

| Label source | Gold-58 macro AUC |
|---|---|
| Qwen3-8B, own prompt (notebook 03) | 0.857 |
| `llm_labels_v4_blend.csv` (stevenleehans) | 0.893 |
| `report_labels_v2.csv` (pilkwang) | 0.863 |
| `report_labels_gpt56sol.csv` (lixin73) | 0.835 |
| Mean of the four, used for training | 0.897 |

Qwen3-8B per finding on gold: ACL 0.884, MCL 0.912, medial meniscus 0.830, lateral meniscus 0.867, medial OA 0.942, lateral OA 0.832, patellofemoral OA 0.868, effusion 0.734, synovitis 0.810, Baker's cyst 0.879, contusion 0.858, fracture 0.869.

### Training runs

Each fold trains on 4/5 of the 4,349 studies without radiologist labels. Validation is the held-out fifth scored against the averaged labels thresholded at 0.5; gold is the 58 radiologist-labelled studies, never trained on. All nine runs are notebook 04 with a different family and folds in its first cell.

| Kernel | Model | Folds | Validation AUC | Gold-58 AUC | Kaggle runtime (T4 x2) |
|---|---|---|---|---|---|
| rsna-radcs-a | RadImageNet ResNet50, spatial pooling | 0, 1 | 0.878, 0.888 | 0.912, 0.916 | 132 min |
| rsna-radcs-b | RadImageNet ResNet50, spatial pooling | 2, 3 | 0.883, 0.882 | 0.918, 0.914 | 141 min |
| rsna-radcs-c | RadImageNet ResNet50, spatial pooling | 4 | 0.878 | 0.908 | 67 min |
| rsna-radcr-a | RadImageNet ResNet50, average pooling | 0, 1 | 0.878, 0.892 | 0.905, 0.908 | 126 min |
| rsna-radcr-b | RadImageNet ResNet50, average pooling | 2, 3 | 0.889, 0.884 | 0.915, 0.913 | 124 min |
| rsna-radcr-c | RadImageNet ResNet50, average pooling | 4 | 0.881 | 0.907 | 65 min |
| rsna-sx50r-a | ImageNet SE-ResNeXt50 | 0, 1 | 0.872, 0.876 | 0.892, 0.876 | 162 min |
| rsna-sx50r-b | ImageNet SE-ResNeXt50 | 2, 3 | 0.879, 0.871 | 0.899, 0.893 | 181 min |
| rsna-sx50r-c | ImageNet SE-ResNeXt50 | 4 | 0.872 | 0.892 | 89 min |

Averaging the five fold models of a family raises gold to 0.920 with spatial pooling, 0.915 with average pooling and 0.906 for SE-ResNeXt50.

## Pipeline

```
RSNA DICOM series and free-text reports (Kaggle competition data)
       |
  setup                DICOM decoder wheels, ImageNet and RadImageNet weights, DICOM header table
       |
  01 eda [CPU]         labels, reports, series and scanner geometry
  02 prepare [CPU]     5 series per study -> 72 slices, 224 px, 140 mm field of view
  03 labels [GPU]      Qwen3-8B reads every report -> 12 probabilities per study
       |
  04 train [GPU]       laterality, label averaging, two folds per run; 9 runs for 3 families x 5 folds
       |
  05 submit [GPU]      15 fold models on the hidden test set, rank average across families
```

Each stage is a Kaggle notebook that reads the earlier stages' saved outputs as inputs. The executed notebooks, with outputs, are in `notebooks/`; the setup kernels are scripts in `setup/`.

| Stage | Kaggle kernel | What it does | Kaggle runtime |
|---|---|---|---|
| Setup | rsna-wheels | Downloads DICOM decoder wheels so the offline notebooks can read JPEG-compressed series | under 1 minute |
| Setup | rsna-weights2 | Saves ImageNet weights for SE-ResNeXt50 and other timm backbones | about 1 minute |
| Setup | rsna-radw | Downloads the official RadImageNet release and converts ResNet50 and DenseNet121 to timm | under 1 minute |
| Setup | rsna-geom | Reads the DICOM headers of every training series: plane, vendor, laterality tag, geometry | 33 minutes |
| 01 EDA | rsna-eda | Label prevalence, report languages, series per study, vendors, laterality tags, field of view | under 1 minute |
| 02 Prepare | rsna-prepv3 | Picks five series per study, sorts slices by position, canonicalises orientation, crops 140 mm, writes a 4,407 x 72 x 224 x 224 array | 46 minutes (38 in its first run) |
| 03 Labels | rsna-labels | Qwen3-8B in float16 reads all 4,407 reports and returns a probability per finding | 119 minutes |
| 04 Train | rsna-radcs-a | Laterality model, label averaging, two folds of the main model with validation and gold scores | 132 minutes |
| 05 Submit | rsna-knee-h3 | Builds the test volumes and rank-averages the 15 fold models | 3 minutes (the scoring run on the hidden test set takes longer) |

Notebooks 01, 02, 03 and 05 were run again on Kaggle with their markdown in place and the same code as the runs behind the submission; notebook 03 first ran as a script and is the same code split into cells. Notebook 04 is the original training run, with the markdown added afterwards, because a new training run would give different weights from the ones in the final submission.

## Why chained notebooks

Kaggle rations GPU time to 30 hours a week, and a submission notebook gets 9 hours and no internet. Preparing the volumes takes 40 to 45 CPU minutes and reading the reports 2 GPU hours, so both run once and every training run reads their saved outputs. Each training run fits two folds in two to three hours, and the submission notebook reads all of them, so a fix to the submission code never costs another training run.

Scoring a submission runs the notebook again on the hidden test set, and that run is not billed against the weekly GPU quota. Some submissions used it to train an extra model on all training studies inside the scoring run.

## Design decisions

**Five series slots.** A study has 3 to 14 series, 5 at the median. Each slot takes the best match for its plane and contrast: sagittal fluid-sensitive (20 slices), sagittal T1 (12), coronal fluid-sensitive (16), coronal T1 (10) and axial fluid-sensitive (14), preferring fat-suppressed series. That makes 72 slices, sampled evenly between 4% and 96% of each series. Slices are sorted by image position along the plane's axis; a check on 1,500 series found that order equal to the slice-normal order in every case.

**Physical crops.** Each slice is rotated and flipped to a canonical orientation from its DICOM direction cosines, cropped to a 140 mm square around the centre using the pixel spacing and resized to 224 px. The field of view is 160 mm at the median and at least 140 mm in about 95% of series, so the crop keeps the joint and puts every study on one physical scale. 192 px at 150 mm scored 0.930 against 0.935; 320 px at 130 mm added 0.004 validation AUC for twice the cost and did not finish inside the 9-hour scoring limit.

**Left knees mirrored.** Findings are named medial and lateral, so every left knee is mirrored to look like a right knee. The DICOM laterality tag is present in half of the studies, mostly from Siemens and Philips scanners. For the rest, a ResNet34 trained in each run on axial slices of the tagged studies decides, together with the knee's position in patient coordinates for scanners where it is reliable; the image model alone is right on about 99% of 353 held-out tagged studies.

**A bag of 3-slice windows.** Training draws 16 random windows per study (5, 2, 4, 2 and 3 from the five slots); each window is three adjacent slices fed as the three input channels. Prediction uses one window centred on every slice, 72 per study. Each window becomes a token with a slot embedding and a position-within-series embedding, two transformer layers mix the tokens, and a separate attention pool per finding chooses which windows matter for it.

**Backbones.** The main backbone is ResNet50 pretrained on 1.35 million radiology images (RadImageNet), with the centre-slice first-layer initialisation described above. Spatial pooling adds a learned attention map over the last feature map next to global average pooling, so a small tear does not vanish in the average. The second family is the same network with average pooling only, the third an ImageNet SE-ResNeXt50.

**Training.** AdamW (weight decay 0.01) with learning rates 2e-4 for the backbone and 1e-3 for the head, one-cycle schedule, 20 epochs, batch 8 across two T4 GPUs in float16, an exponential moving average of the weights (decay 0.998), and random rotation, scale, shift and intensity changes. The loss is binary cross-entropy against the averaged soft labels.

**Rank averaging.** The fold models of one family are averaged as probabilities; families are combined by averaging per-finding ranks, which is what ROC AUC rewards.

**Things that did not help.** Distilled labels, a fifth and sixth label source, reversed-slice test-time augmentation, stratified window sampling, a fourth model trained in the scoring run, DenseNet121 (failed inside the scoring run only), DINOv2-S (did not fine-tune in this recipe), CoAtNet and EfficientNet (crashed or ran 7 times slower on the T4).

## Reproduce

1. Join the competition and authenticate the Kaggle CLI on a phone-verified account (needed for GPU, and for internet in the setup kernels).
2. In every `kernel-metadata.json`, replace `deveshupathak` with your username.
3. Run the setup kernels, for example:
   ```bash
   kaggle kernels push -p setup/wheels
   ```
   Do the same for `setup/weights`, `setup/radimagenet` and `setup/headers`. `setup/radimagenet` downloads the official RadImageNet release from Google Drive and loads its whole-model pickles with `torch.load(weights_only=False)`, which can execute code embedded in the file; run it only in an isolated environment such as a Kaggle kernel, as here. Two inputs were private datasets: `rsna-headers` is a column subset of the `geom.csv` written by `setup/headers` (attaching that kernel instead works the same), and `rsna-timm-weights` holds `resnet34.pt`, the state dict of `timm.create_model("resnet34", pretrained=True, num_classes=0)`, used by the laterality model.
4. Push notebooks 01 to 03 in order with `kaggle kernels push -p notebooks/<stage>`, waiting for each to finish (`kaggle kernels status`).
5. Run notebook 04 nine times under the kernel names that notebook 05 reads (rsna-radcs-a to rsna-sx50r-c), setting `FAM` to `radcs`, `radcr` or `sx50r` and `RUN_FOLDS` to `[0, 1]`, `[2, 3]` or `[4]` in its first cell. Its inputs rsna-own-a and rsna-own-b are earlier models read only by the final check cell and can be removed from the metadata.
6. Push notebook 05, then submit it from its Kaggle page. The scoring run builds the test volumes and predicts the hidden test set.

## Repository layout

```
notebooks/   the five executed Kaggle notebooks, each with its kernel metadata
setup/       the four setup scripts that ran on Kaggle, each with its kernel metadata
```

## Data, models and licences

None of the competition data is redistributed in this repository; the notebooks read it on Kaggle and print only aggregate statistics.

1. Competition data: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), Radiological Society of North America, used under the competition rules.
2. Public label tables, all CC0 1.0: [stevenleehans/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang/rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), [lixin73/rsna-knee-llm-report-labels-sol56](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56).
3. Qwen3-8B (Apache 2.0) from [Kaggle Models](https://www.kaggle.com/models/qwen-lm/qwen-3): Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388, 2025.
4. RadImageNet (official release, MIT licence): X. Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning", Radiology: Artificial Intelligence 4(5), 2022.
5. ImageNet weights from timm (Apache 2.0): R. Wightman, PyTorch Image Models, 2019. ResNet: K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition", CVPR 2016. SE-ResNeXt: J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks", CVPR 2018; S. Xie et al., "Aggregated Residual Transformations for Deep Neural Networks", CVPR 2017.
