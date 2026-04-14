import argparse

from kb import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地医学知识库索引")
    parser.add_argument("--resources-dir", default="resources")
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    count = build_index(
        resources_dir=args.resources_dir,
        persist_directory=args.db_dir,
        reset=args.reset,
    )
    print(f"已写入 {count} 个 chunk 到 {args.db_dir}")


if __name__ == "__main__":
    main()
