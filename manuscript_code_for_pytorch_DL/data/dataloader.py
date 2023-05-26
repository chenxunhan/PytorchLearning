from torch.utils.data import DataLoader

def get_my_dataloader(dataset, batch_size=32, do_shuffle=True, num_workers=8):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=do_shuffle,
                      num_workers=num_workers
                      )
