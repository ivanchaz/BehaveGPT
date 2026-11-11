from pathlib import Path

cache_dir = Path("temp")
special_tokens = ["<PAD>"]


def __get_cache(file_name: str) -> map[str, int]:
    entries = {}
    vocabulary_file = Path(cache_dir, file_name)
    vocabulary_file.touch()

    with open(vocabulary_file) as file:
        content = file.readlines()

        for line in content:
            parts = line.split(":")
            key, value = parts[0].strip(), parts[1].strip()

            entries[key] = int(value)

    for token in special_tokens:
        if token in entries:
            continue

        entries[token] = len(entries)

    return entries 

def __persist_vocabularies(file_name: str, words: list[str]) -> None:
    cache = __get_cache(file_name)
    keys = cache.keys()
    max_index = (
        max([val for val in cache.values()])
        if len(cache) != 0 else
        0
    )

    new_entries = {}
    for word in words:
        if word in keys:
            continue

        new_entries[word] = max_index
        max_index += 1

    lines = []
    for key, value in new_entries.items():
        lines.append(f"{key}:{value}")
        lines.append("\n")
    lines = lines[:-1]

    file_path = Path(cache_dir, file_name)
    with open(file_path, mode="a") as file:
        file.writelines(lines)

def get_event_vocabularies() -> map[str, int]:
    return __get_cache("vocabulary") 

def get_spatial_vocabularies() -> map[str, int]:
    return __get_cache("spatial")

def persist_event_vocabularies(words: list[str]) -> None:
    __persist_vocabularies("vocabulary", words)

def persist_spatial_vocabularies(words: list[str]) -> None:
    __persist_vocabularies("spatial", words)
