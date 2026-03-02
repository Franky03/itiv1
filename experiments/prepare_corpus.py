import bz2
import pathlib
import urllib.request

CORPUS_DIR = pathlib.Path("corpus/silesia")

# URLs dos arquivos (verifique no site se mudaram)
SILESIA_FILES = [
    ("dickens",  "http://sun.aei.polsl.pl/~sdeor/corpus/dickens.bz2"),
    ("mozilla",  "http://sun.aei.polsl.pl/~sdeor/corpus/mozilla.bz2"),
    ("mr",       "http://sun.aei.polsl.pl/~sdeor/corpus/mr.bz2"),
    ("nci",      "http://sun.aei.polsl.pl/~sdeor/corpus/nci.bz2"),
    ("ooffice",  "http://sun.aei.polsl.pl/~sdeor/corpus/ooffice.bz2"),
    ("osdb",     "http://sun.aei.polsl.pl/~sdeor/corpus/osdb.bz2"),
    ("reymont",  "http://sun.aei.polsl.pl/~sdeor/corpus/reymont.bz2"),
    ("samba",    "http://sun.aei.polsl.pl/~sdeor/corpus/samba.bz2"),
    ("sao",      "http://sun.aei.polsl.pl/~sdeor/corpus/sao.bz2"),
    ("webster",  "http://sun.aei.polsl.pl/~sdeor/corpus/webster.bz2"),
    ("xml",      "http://sun.aei.polsl.pl/~sdeor/corpus/xml.bz2"),
    ("x-ray",    "http://sun.aei.polsl.pl/~sdeor/corpus/x-ray.bz2"),
]


def download_corpus():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in SILESIA_FILES:
        dest = CORPUS_DIR / name
        if dest.exists():
            print(f"  {name:12s}: já existe ({dest.stat().st_size:,} bytes)")
            continue

        print(f"  {name:12s}: baixando...", end=" ", flush=True)
        bz2_path = CORPUS_DIR / f"{name}.bz2"

        try:
            urllib.request.urlretrieve(url, bz2_path)
            raw = bz2.decompress(bz2_path.read_bytes())
            dest.write_bytes(raw)
            bz2_path.unlink()
            print(f"OK ({len(raw):,} bytes)")
        except Exception as e:
            print(f"ERRO: {e}")

    print(f"\nCorpus em: {CORPUS_DIR.resolve()}")
    total = sum(f.stat().st_size for f in CORPUS_DIR.iterdir() if f.is_file())
    print(f"Total: {total:,} bytes ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    download_corpus()