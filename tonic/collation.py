try:
    import torch
except:
    ...


class PadTensors:
    """This is a custom collate function for a pytorch dataloader to load multiple
    event recordings at once. It's intended to be used in combination with sparse tensors.
    All tensor sizes are extended to the largest one in the batch, i.e. the longest recording.

    Example:
        >>> dataloader = torch.utils.data.DataLoader(dataset,
        >>>                                          batch_size=10,
        >>>                                          collate_fn=tonic.collation.PadTensors(),
        >>>                                          shuffle=True)

    """

    def __init__(self, batch_first: bool = False):
        self.batch_first = batch_first

    def __call__(self, batch):
        samples_output = []
        targets_output = []

        max_length = max([sample.shape[0] for sample, target in batch])
        for sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            samples_output.append(
                torch.cat(
                    (
                        sample,
                        torch.zeros(max_length - sample.shape[0], *sample.shape[1:]),
                    )
                )
            )
            targets_output.append(target)
        return (
            torch.stack(samples_output, 0 if self.batch_first else 1),
            torch.tensor(targets_output),
        )
