from meshmind.train import TrainConfig, smoke_train


def test_cpu_smoke_training_learns():
    first, last = smoke_train(TrainConfig(steps=12, batch_size=2, sequence_length=16))
    assert last < first
