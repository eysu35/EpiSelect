import dataclasses
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
from transformers import AutoTokenizer

# changing any constant below requires bumping VERSION and rebuilding
TOKEN_CAP = 4_000_000_000  # max tokens per domain
VAL_FRACTION = 0.01
SHUFFLE_SEED = 0
SHUFFLE_BUFFER = 10_000
TOKENIZE_BATCH = 64

# one built split; val/train are fixed slices of the shuffled stream
VAL_SPLIT = f"train[:{round(VAL_FRACTION * 100)}%]"
TRAIN_SPLIT = f"train[{round(VAL_FRACTION * 100)}%:]"
SPLITS = {"train": TRAIN_SPLIT, "val": VAL_SPLIT}
SPLIT_FRACTION = {"train": 1 - VAL_FRACTION, "val": VAL_FRACTION}
TOKEN_COUNTS_FILE = "token_counts.json"  # exact per-split token counts, written next to the built shards


def token_counts_path(builder):
    return os.path.join(builder.data_dir, TOKEN_COUNTS_FILE)


def count_tokens(builder):
    # split.num_bytes is the varint-serialized size (~2 bytes/token), so count for real by reading the shards
    counts = {}
    for name, split in SPLITS.items():
        ds = builder.as_dataset(split=split).map(lambda x: tf.shape(x["tokens"])[0], num_parallel_calls=tf.data.AUTOTUNE)
        counts[name] = int(ds.reduce(tf.constant(0, tf.int64), lambda acc, n: acc + tf.cast(n, tf.int64)).numpy())
    counts["num_examples"] = int(builder.info.splits["train"].num_examples)
    return counts


def write_token_counts(builder):
    counts = count_tokens(builder)
    with tf.io.gfile.GFile(token_counts_path(builder), "w") as f:
        json.dump(counts, f, indent=2)
    return counts


def read_token_counts(builder):
    path = token_counts_path(builder)
    if not tf.io.gfile.exists(path):
        raise RuntimeError(f"{path} missing -- run scripts/build_commonpile_tfds.py --count_only for this domain")
    with tf.io.gfile.GFile(path) as f:
        return json.load(f)

# domain -> (hf dataset, pinned revision)
DOMAINS = {
    "arxiv_abstracts":               ("common-pile/arxiv_abstracts_filtered",               "dc1ceab4755eb037ec61e49cf1350dab7ceee6e7"),
    "arxiv_papers":                  ("common-pile/arxiv_papers_filtered",                  "033cf7f53f9b348deec868c1a5a48484f3ee9e52"),
    "biodiversity_heritage_library": ("common-pile/biodiversity_heritage_library_filtered", "0486ed637d0d7aaff264bc77fe21a7444e0215cd"),
    "caselaw":                       ("common-pile/caselaw_access_project_filtered",        "50e1961a5ed8fdddcee64c9e66e1338de8953b63"),
    "cc_common_crawl":               ("common-pile/cccc_filtered",                          "03a3de5713a0bb23267d26724346508af0f25327"),
    "data_provenance":               ("common-pile/data_provenance_initiative_filtered",    "8f5afcf585e4618ff12f5b6d49cb5242faf5afbd"),
    "doab":                          ("common-pile/doab_filtered",                          "defb24ca72ef6aba6ce0228b669eec06dcfbffbc"),
    "foodista":                      ("common-pile/foodista_filtered",                      "bf2c7aaa2d32be1e7d4d4779a6017407236831de"),
    "github":                        ("common-pile/github_archive_filtered",                "52282fe96670254bdc0d44dd718ee7a27210ee85"),
    "gutenberg":                     ("common-pile/project_gutenberg_filtered",             "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7"),
    "library_of_congress":           ("common-pile/library_of_congress_filtered",           "56725c7aa1bb320703e22eb5f42903173d5bac3d"),
    "libretexts":                    ("common-pile/libretexts_filtered",                    "70388bca52b4a93515e14b1d56618fd7944988fd"),
    "news":                          ("common-pile/news_filtered",                          "59aaa8f104e189e6fb8033f0ed319c5c343a03b1"),
    "oercommons":                    ("common-pile/oercommons_filtered",                    "506b6159dadcbc0dc67611cea024eedb04232fb2"),
    "pes2o":                         ("common-pile/peS2o_filtered",                         "297747513bfb0ff1fbf61ddad3b03319d0f04597"),
    "pre_1929_books":                ("common-pile/pre_1929_books_filtered",                "23f9d96dbb1db3324bbc9fbfe1f8299cc799c4d1"),
    "pressbooks":                    ("common-pile/pressbooks_filtered",                    "1a1d3b50d77f834370f8eb4c0d174668dd1676bb"),
    "public_domain_review":          ("common-pile/public_domain_review_filtered",          "efc7f21259d069a27d6ca3655f74fda969f82b01"),
    "pubmed":                        ("common-pile/pubmed_filtered",                        "c156f0569a92d8f2edc33cebe1f72f7d3e1cae84"),
    "python_peps":                   ("common-pile/python_enhancement_proposals_filtered",  "582170907dd303c207770fceacd38e6abf133edc"),
    "regulations":                   ("common-pile/regulations_filtered",                   "3327364490dfc7929009226ad667eceb2441d93a"),
    "stackexchange":                 ("common-pile/stackexchange_filtered",                 "c0ac7373830c688a43fc12d1988c4b19ccd884ab"),
    "stackv2_edu":                   ("common-pile/stackv2_edu_filtered",                   "c354dbe88469a1153e97c6a63ac50591849654de"),
    "stackv2_html":                  ("common-pile/stackv2_html_filtered",                  "92c9fa898d6af643db9c7d73e75219f862be8ca0"),
    "ubuntu_irc":                    ("common-pile/ubuntu_irc_filtered",                    "84f88c986584f11d672befab542fa4d5123f3e8f"),
    "uk_hansard":                    ("common-pile/uk_hansard_filtered",                    "c88adc44309aa255a41b51cef93ba783f775fe23"),
    "usgpo":                         ("common-pile/usgpo_filtered",                         "b150cc22211de4d57f1b7f570097a00e65042424"),
    "uspto":                         ("common-pile/uspto_filtered",                         "13894c5462467c843163693269d9266ec2c772b4"),
    "wikimedia":                     ("common-pile/wikimedia_filtered",                     "0641bb84bd9b7162bcddf8be7836822161a9a342"),
    "wikiteam":                      ("common-pile/wikiteam_filtered",                      "f4ed055b57763a8f12238824140914b9eb098cab"),
    "youtube":                       ("common-pile/youtube_filtered",                       "dff8c8a54e98bce64c2e7ce9a8466c144c1cddd6"),
}
TASKS = list(DOMAINS)


def tokenize_capped(texts, tokenize, cap, batch=TOKENIZE_BATCH):
    # yields (idx, tokens) from a text stream, stopping once cap tokens have been emitted
    total, idx, buf = 0, 0, []

    def flush():
        nonlocal total, idx
        for tokens in tokenize(buf):
            yield idx, tokens
            idx += 1
            total += len(tokens)
            if total >= cap:
                return
        buf.clear()

    for text in texts:
        if not text:
            continue
        buf.append(text)
        if len(buf) == batch:
            yield from flush()
            if total >= cap:
                return
    if buf:
        yield from flush()


@dataclasses.dataclass
class CommonPileConfig(tfds.core.BuilderConfig):
    hf_name: str = ""
    revision: str = ""


_CONFIGS = [CommonPileConfig(name=task, hf_name=hf, revision=rev, description=task) for task, (hf, rev) in DOMAINS.items()]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


class CommonPileGrouped(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("3.0.0")  # 3.0.0: shuffled stream, token-counted 4B cap, val by slicing
    BUILDER_CONFIGS = _CONFIGS

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "tokens": tfds.features.Tensor(shape=(None,), dtype=tf.int32),
            }),
            supervised_keys=("tokens", "tokens"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        cfg = self.builder_config
        return [tfds.core.SplitGenerator(name="train", gen_kwargs={"hf_name": cfg.hf_name, "revision": cfg.revision})]

    def _generate_examples(self, hf_name: str, revision: str):
        # shuffle reorders the shards and buffers docs, so the capped prefix is a random sample
        ds = load_dataset(hf_name, split="train", streaming=True, revision=revision)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
        texts = (doc.get("text", "") for doc in ds)
        tokenize = lambda batch: tokenizer(batch)["input_ids"]  # noqa: E731
        for idx, tokens in tokenize_capped(texts, tokenize, TOKEN_CAP):
            yield idx, {"tokens": tokens}
