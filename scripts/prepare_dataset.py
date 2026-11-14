import argparse
import _bootstrap
from src.data.prepare_vqa import prepare_fixture, prepare_official

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--root", default=str(_bootstrap.ROOT))
    args = parser.parse_args()
    print(prepare_official(args.root) if args.official else prepare_fixture(args.root))
