import torch
from tqdm import tqdm
import torch.nn.functional as F


train_acc = []
train_loss = []
test_acc = []
test_losses = []

def train(model, device, train_loader, optimizer, epoch):

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            y_pred = model(data)

            loss = F.nll_loss(y_pred, target, reduction='sum').item()
            pred = y_pred.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

            test_acc.append(100. * correct / len(test_loader.dataset))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))