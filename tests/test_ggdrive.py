from ggdrive import ggdrive


def test_crawl_files():
    results = ggdrive.ls()
    print("")
    print(f"Got {len(results)} files in GGDrive")
    for file in results:
        print(f"  - {file=}")
