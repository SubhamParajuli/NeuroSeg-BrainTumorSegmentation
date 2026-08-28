from brain_seg.dataset import get_image_mask_pairs


DATA_DIR = "data/raw"


def test_dataset_pairs():

    pairs = get_image_mask_pairs(DATA_DIR)

    assert len(pairs) > 0

    for image_path, mask_path in pairs[:10]:

        assert image_path.endswith(".tif")
        assert mask_path.endswith("_mask.tif")