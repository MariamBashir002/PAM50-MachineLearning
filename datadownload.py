import os
from urllib.request import urlretrieve

os.makedirs("data", exist_ok=True)

FILES = {
    "data/HiSeqV2.gz": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "data/BRCA_clinicalMatrix.tsv": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
}

for path, url in FILES.items():
    if os.path.exists(path):
        print("already have", path)
        continue
    print("downloading", path)
    urlretrieve(url, path)
