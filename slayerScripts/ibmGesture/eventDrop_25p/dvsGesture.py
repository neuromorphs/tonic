from datetime import datetime
import numpy as np

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import slayerSNN as snn
from slayerSNN.learningStats import learningStats
import slayerSNN.optimizer as optim

import spike_data_augmentation
import spike_data_augmentation.transforms as transforms


# Define dataset module
class IBMGestureDataset(Dataset):
    def __init__(self, samplingTime, sampleLength, train=False, transform=None):
        self.samplingTime = samplingTime
        self.sampleLength = sampleLength
        self.train = train
        self.nTimeBins = int(sampleLength / samplingTime)
        self.dataset = spike_data_augmentation.datasets.IBMGesture(
            save_to="../data", train=train, download=False, transform=transform
        )

    def __getitem__(self, index):
        events, label = self.dataset[index]
        inputSpikes = snn.io.event(
            events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        ).toSpikeTensor(
            torch.zeros((2, 128, 128, self.nTimeBins)),
            samplingTime=self.samplingTime,
            randomShift=self.train,
        )
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((11, 1, 1, 1))
        desiredClass[label, ...] = 1

        return inputSpikes, desiredClass, label

    def __len__(self):
        return len(self.dataset)


# Define the network
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams["neuron"], netParams["simulation"])
        self.slayer = slayer
        # define network functions
        self.conv1 = torch.nn.utils.weight_norm(
            slayer.conv(2, 16, 5, padding=2), name="weight"
        )
        self.conv2 = torch.nn.utils.weight_norm(
            slayer.conv(16, 32, 3, padding=1), name="weight"
        )
        self.fc1 = torch.nn.utils.weight_norm(
            slayer.dense((8 * 8 * 32), 512), name="weight"
        )
        self.fc2 = torch.nn.utils.weight_norm(slayer.dense(512, 11), name="weight")
        self.pool1 = slayer.pool(4)
        self.pool2 = slayer.pool(2)
        self.pool3 = slayer.pool(2)
        self.delay1 = slayer.delay(16)
        self.delay2 = slayer.delay(32)
        self.delay3 = slayer.delay(512)

    def forward(self, spikeInput):
        spike = self.slayer.spike(self.pool1(self.slayer.psp(spikeInput)))  # 32, 32, 2

        spike = self.slayer.spike(self.conv1(self.slayer.psp(spike)))  # 32, 32, 16
        spike = self.delay1(spike)

        spike = self.slayer.spike(self.pool2(self.slayer.psp(spike)))  # 16, 16, 16

        spike = self.slayer.spike(self.conv2(self.slayer.psp(spike)))  # 16, 16, 32
        spike = self.delay2(spike)

        spike = self.slayer.spike(self.pool3(self.slayer.psp(spike)))  #  8,  8, 32
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))

        spike = self.slayer.spike(self.fc1(self.slayer.psp(spike)))  # 512
        spike = self.delay3(spike)

        spike = self.slayer.spike(self.fc2(self.slayer.psp(spike)))  # 11

        return spike

    def clamp(self):
        self.delay1.delay.data.clamp_(0)
        self.delay2.delay.data.clamp_(0)
        self.delay3.delay.data.clamp_(0)

    def gradFlow(self, path):
        gradNorm = lambda x: torch.norm(x).item() / torch.numel(x)

        grad = []
        grad.append(gradNorm(self.conv1.weight_g.grad))
        grad.append(gradNorm(self.conv2.weight_g.grad))
        grad.append(gradNorm(self.fc1.weight_g.grad))
        grad.append(gradNorm(self.fc2.weight_g.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()


if __name__ == "__main__":
    netParams = snn.params("network.yaml")

    # Define the cuda device to run the code on.
    device = torch.device("cuda:2")

    # Create network instance.
    net = Network(netParams).to(device)

    # Create snn loss instance.
    error = snn.loss(netParams).to(device)

    netParamsTest = snn.params("network.yaml")
    netParamsTest["training"]["error"]["tgtSpikeRegion"]["start"] = 0
    netParamsTest["training"]["error"]["tgtSpikeRegion"]["stop"] = 1500
    netParamsTest["training"]["error"]["tgtSpikeCount"][False] *= (
        1500 / netParams["simulation"]["tSample"]
    )
    netParamsTest["training"]["error"]["tgtSpikeCount"][True] *= (
        1500 / netParams["simulation"]["tSample"]
    )

    errorTest = snn.loss(netParamsTest).to(device)

    # Define optimizer module.
    optimizer = optim.Nadam(net.parameters(), lr=0.01)

    # Dataset and dataLoader instances.
    trainingSet = IBMGestureDataset(
        samplingTime=netParams["simulation"]["Ts"],
        sampleLength=netParams["simulation"]["tSample"],
        train=True,
        transform=transforms.Compose(
            [transforms.DropEvent(drop_probability=0.25, random_drop_probability=True)]
        ),
    )

    testingSet = IBMGestureDataset(
        samplingTime=netParams["simulation"]["Ts"], sampleLength=1500, train=False
    )

    trainLoader = DataLoader(
        dataset=trainingSet, batch_size=8, shuffle=True, num_workers=1
    )
    testLoader = DataLoader(
        dataset=testingSet, batch_size=8, shuffle=True, num_workers=1
    )

    # Learning stats instance.
    stats = learningStats()

    # lastEpoch = stats.load('Trained/', modulo=10)
    # checkpoint = torch.load('Logs/checkpoint%d.pt'%(lastEpoch -1))
    # net.module.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # Visualize the input spikes (first five samples).
    # for i in range(5):
    #   input, target, label = trainingSet[i]
    #   snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 128, 128, -1)).cpu().data.numpy()), repeat=True)

    for epoch in range(1000):
        # for epoch in range(lastEpoch, 2000):
        tSt = datetime.now()

        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            net.train()

            # Move the input and target to correct GPU.
            input = input.to(device)
            target = target.to(device)

            # Forward pass of the network.
            output = net.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            stats.training.numSamples += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            # Reset gradients to zero.
            optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Clamp delays
            net.clamp()

            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
        for i, (input, target, label) in enumerate(testLoader, 0):
            net.eval()
            with torch.no_grad():
                input = input.to(device)
                target = target.to(device)

                output = net.forward(input)

                stats.testing.correctSamples += torch.sum(
                    snn.predict.getClass(output) == label
                ).data.item()
                stats.testing.numSamples += len(label)

                loss = (
                    errorTest.numSpikes(output, target)
                    * netParams["simulation"]["tSample"]
                    / 1500
                )
                stats.testing.lossSum += loss.cpu().data.item()

            stats.print(epoch, i)

        # Update stats.
        stats.update()
        stats.plot(saveFig=True, path="Trained/")
        net.gradFlow(path="Trained/")
        if stats.testing.bestAccuracy is True:
            torch.save(net.state_dict(), "Trained/dvsGesture.pt")

        if epoch % 10 == 0:
            torch.save(
                {"net": net.state_dict(), "optimizer": optimizer.state_dict()},
                "Logs/checkpoint%d.pt" % epoch,
            )

        stats.save("Trained/")
