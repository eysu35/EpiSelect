import argparse
import os
import subprocess

import tensorflow as tf
import tensorflow_datasets as tfds

from src.tfds import common_pile_grouped


def sync(local_dir, bucket_dir):
    print(f"Syncing {local_dir} -> {bucket_dir}", flush=True)
    subprocess.run(["gcloud", "storage", "rsync", "-r", local_dir, bucket_dir], check=True)


def build(domain, data_dir, bucket=None, count_only=False):
    builder = tfds.builder(f"common_pile_grouped/{domain}", data_dir=data_dir)
    if not count_only:
        print(f"Building common_pile_grouped/{domain} -> {data_dir}", flush=True)
        builder.download_and_prepare()
    if tf.io.gfile.exists(common_pile_grouped.token_counts_path(builder)):
        counts = common_pile_grouped.read_token_counts(builder)
    else:
        print(f"Counting tokens in {domain}...", flush=True)
        counts = common_pile_grouped.write_token_counts(builder)
    print(f"Done: {domain}: {counts['num_examples']:,} docs, {counts['train']:,} train + {counts['val']:,} val tokens", flush=True)
    if bucket:
        rel = os.path.join("common_pile_grouped", domain, str(builder.version))
        sync(os.path.join(data_dir, rel), os.path.join(bucket, rel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", help="comma-separated subset of domains (default: all)")
    parser.add_argument("--list", action="store_true", help="print the domain list, one per line, and exit")
    parser.add_argument("--data_dir", help="local build dir (or gs:// to build straight into a bucket)")
    parser.add_argument("--sync", metavar="GS_DIR", help="after each domain, rsync it to this gs:// data dir")
    parser.add_argument("--count_only", action="store_true", help="skip building; write token_counts.json for already-built domains")
    args = parser.parse_args()

    if args.list:
        print("\n".join(common_pile_grouped.TASKS))
        return
    assert args.data_dir, "--data_dir is required"

    domains = args.domains.split(",") if args.domains else common_pile_grouped.TASKS
    unknown = set(domains) - set(common_pile_grouped.TASKS)
    assert not unknown, f"Unknown domains: {sorted(unknown)}. Valid: {common_pile_grouped.TASKS}"

    for domain in domains:
        build(domain, args.data_dir, args.sync, args.count_only)


if __name__ == "__main__":
    main()
