from utils.ggdrive import ggdrive


def test_import():
    pass


def test_crawl_files():
    results = ggdrive.ls()
    print(f"Got {len(results)} files in GGDrive")
    for file in results:
        print(f"  - {file.name=}, {file.id=}, {file.url}")
