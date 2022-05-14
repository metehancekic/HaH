import torch
import numpy as np
from torch import nn


def test_noisy(model, test_loader, noise_std, noise_type="gaussian"):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    snrs = torch.zeros(len(model.layer_inputs)).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, list):
                data = data[0]
                target = target[0]

            data, target = data.to(device), target.to(device)
            output = model(data)

            clean_layer_outs = list(model.layer_inputs.values())

            # noise = torch.normal(mean=torch.zeros_like(
            #     data), std=noise_std*torch.ones_like(data))

            # noisy_data = (data+noise).clamp(0.0, 1.0)
            noisy_data = get_noisy_images(data, noise_std, noise_type)

            output = model(noisy_data)

            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item() * data.size(0)

            # breakpoint()
            noisy_layer_outs = list(model.layer_inputs.values())
            for layer_no, _ in enumerate(noisy_layer_outs):
                snrs[layer_no] += (torch.norm(clean_layer_outs[layer_no], p=2, dim=(1, 2, 3))/(torch.norm(noisy_layer_outs[layer_no]-clean_layer_outs[layer_no], p=2,dim=(1, 2, 3))+0.000000001)).square().sum()

    test_size = len(test_loader.dataset)
    test_acc = test_correct / test_size
    snrs /= test_size
    snrs = snrs.tolist()
    snrs = [10*np.log10(s) for s in snrs]


    return test_acc, test_loss/test_size, snrs

def test_small_noisy(model, test_loader, noise_std=0.1):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    snrs = torch.zeros(len(model.layer_inputs)).to(device)
    with torch.no_grad():
        data, target = test_loader.__iter__().__next__()

        data, target = data.to(device), target.to(device)
        output = model(data)

        clean_layer_outs = list(model.layer_inputs.values())

        noise = torch.normal(mean=torch.zeros_like(
            data), std=noise_std*torch.ones_like(data))

        noisy_data = (data+noise).clamp(0.0, 1.0)
        output = model(noisy_data)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()
        
    print(f"Noisy Accuracy: {test_correct/target.shape[0]}")

def get_noisy_images(data, noise_std, noise_type="gaussian"):
    
    if noise_type=="gaussian":
        noise = torch.normal(mean=torch.zeros_like(
            data), std=noise_std*torch.ones_like(data))
    elif noise_type=="uniform":
        lims = noise_std * (3.0)**0.5
        noise = torch.zeros_like(data).uniform_(-lims, lims)
    else:
        raise NotImplementedError

    noisy_data = (data+noise).clamp(0.0, 1.0)

    return noisy_data


