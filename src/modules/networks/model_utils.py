import os

import torch


def create_dataloader(data, labels, batch_size=128, shuffle=False,
                      drop_last=True, num_workers=8):
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(data), torch.Tensor(labels))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last)
    return dataloader


def train(net, trainloader, valloader, epochs, optimizer, criterion,
          lr_scheduler=None, early_patience=10, path=None, regression=False,
          verbose=0):
    device = next(net.parameters()).device
    best_loss, early_stopping = None, 0
    if path is None:
        path = 'tmp_model.pt'
    for epoch in range(epochs):
        # train
        net.train()
        train_loss, correct, total = 0, 0, 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            if not regression:
                loss = criterion(outputs, labels.long())
            else:
                outputs_flat = outputs.view(outputs.size()[0], -1)
                labels_flat = labels.view(labels.size()[0], -1)
                loss = criterion(outputs_flat, labels_flat)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if not regression:
                _, predicted = torch.max(outputs, 1)
                total += labels.long().size(0)
                correct += (predicted == labels).sum().item()
        # eval
        net.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                if not regression:
                    loss = criterion(outputs, labels.long())
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.long().size(0)
                    val_correct += (predicted == labels).sum().item()
                else:
                    outputs_flat = outputs.view(outputs.size()[0], -1)
                    labels_flat = labels.view(labels.size()[0], -1)
                    loss = criterion(outputs_flat, labels_flat)
                val_loss += loss.item()
            if verbose:
                if not regression:
                    print(('Epoch: %d/%d | Loss: %.4f | Accuracy: %.2f %% | ' +
                          'ValLoss: %.4f | ValAccuracy: %.2f %%') %
                          (epoch+1, epochs, train_loss / len(trainloader),
                           100 * correct / total, val_loss / len(valloader),
                           100 * val_correct / val_total))
                else:
                    print('Epoch: %d/%d | Loss: %.4f | ValLoss: %.4f' %
                          (epoch+1, epochs, train_loss / len(trainloader),
                           val_loss / len(valloader)))
            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                early_stopping = 0
                torch.save(net.state_dict(), path)
            else:
                early_stopping += 1
            if early_patience > 0 and early_stopping > early_patience:
                break
    net.load_state_dict(torch.load(path))
    if path == 'tmp_model.pt':
        os.remove(path)


def predict(net, data, batch_size=32):
    device = next(net.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            input_x = torch.Tensor(data[i:i + batch_size])
            out = net(input_x.to(device))
            out = out.detach().to('cpu')
            out = torch.argmax(out, 1)
            preds.append(out)
    preds = torch.cat(preds)
    return preds
