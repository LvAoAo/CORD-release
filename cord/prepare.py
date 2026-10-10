"""Create text-only episode placeholders without downloading unrelated data."""
from pathlib import Path


def prepare(directory: Path, train_size: int, validation_size: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    for split, size in (("train", train_size), ("test", validation_size)):
        rows = [{"data_source": "text", "prompt": [{"role": "user", "content": ""}],
                 "ability": "agent", "extra_info": {"split": split, "index": i}}
                for i in range(size)]
        pq.write_table(pa.Table.from_pylist(rows), directory / f"{split}.parquet")
