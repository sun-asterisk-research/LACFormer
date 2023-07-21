from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class PolypDataset(CustomDataset):
    CLASSES = ("Background", "Polyp")
    PALETTE = [0, 1]
    def __init__(self, img_suffix, seg_map_suffix, **kwargs):
        super(PolypDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        assert self.file_client.exists(self.img_dir)