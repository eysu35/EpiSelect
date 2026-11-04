import argparse
import subprocess

from src.eval import OOD_SOURCES, _get_ood_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True, help="local ood_cache_dir (see configs/eval.yaml)")
    parser.add_argument("--sync", metavar="GS_DIR", help="rsync the cache dir to this gs:// dir when done")
    args = parser.parse_args()

    for name in OOD_SOURCES:
        arr = _get_ood_tokens(name, args.cache_dir)
        print(f"{name}: {len(arr):,} tokens", flush=True)
    if args.sync:
        subprocess.run(["gcloud", "storage", "rsync", "-r", args.cache_dir, args.sync], check=True)


if __name__ == "__main__":
    main()
