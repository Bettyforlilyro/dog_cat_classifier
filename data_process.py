from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

ms_train_dataset = MsDataset.load(dataset_name='cats_and_dogs', namespace='tany0699', subset_name='default', split='train', trust_remote_code=True)

print(next(iter(ms_train_dataset)))

ms_val_dataset = MsDataset.load(dataset_name='cats_and_dogs', namespace='tany0699', subset_name='default', split='validation', trust_remote_code=True)
print(next(iter(ms_val_dataset)))
