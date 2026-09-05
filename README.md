# Neural Module Networks, reimplemented

Summer research at the **UCLA Visual Intelligence Lab** (June to August 2024), under a graduate mentor.

This repository holds a PyTorch reimplementation of Andreas et al., *[Deep Compositional Question Answering with Neural Module Networks](https://arxiv.org/abs/1511.02799)* (CVPR 2016), alongside the deep learning coursework I worked through that summer to get there.

---

## Why this was harder than it sounds

Neural Module Networks answer a visual question by **assembling a network per question**. "What color is the cat left of the sofa?" is parsed into a layout, and that layout is instantiated from a small library of reusable neural modules (attend, re-attend, classify, combine) whose weights are shared across every question that uses them. The architecture is different for every input, which is an awkward thing to express in a framework built around a fixed graph and batched tensors.

The published implementation targets a deep learning framework that has since been abandoned, so it does not run on anything current. Reproducing the paper meant reading it closely, reading code that no longer executes, and rebuilding the pieces against PyTorch semantics rather than translating line by line.

## Layout

```
Neural_Module_Networks/
├── src/                          ← my reimplementation
│   ├── models/attention/
│   │   ├── forward_lstm.py       question encoder
│   │   └── forward_image_data.py image feature pathway
│   ├── data/                     COCO images and .mat attention annotations
│   └── utils/io.py               loaders for the original .mat format
│
└── nmn2-master/                  ← the authors' original code, vendored
                                    unmodified for reference. Not mine,
                                    and it does not run as-is.

dl_learning/                      ← the ground work
├── dl_textbook/                  chapter exercises, implemented from scratch
└── blog_learning/                deep-learning-from-scratch walkthroughs
```

**`Neural_Module_Networks/src/` is my work.** `nmn2-master/` is the original repository, kept in place so the two can be read side by side.

## Data

Attention supervision comes from the paper's `.mat` annotation files paired with COCO `val2014` images. `src/utils/io.py` handles the loading, since that format predates most of what you would reach for today.

## Running it

```bash
pip install -r dl_learning/requirements.txt
python -m src.models.attention.forward_image_data
```

## Status

A research replication, not a library. It is here as a record of the work, and because reimplementing a paper whose reference code has rotted is a genuinely useful exercise that not enough people do.
