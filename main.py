# Importing Libraries
import argparse
import time
import copy
import os
from os import path
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import json

# Custom Libraries
import utils
import distilled_datasets

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Main
def main(args, ITE=0, replication=0):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        distilled_traindataset = distilled_datasets.CustomDataset(data_path=args.data_path)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
    elif args.dataset == "cifar10_distilled":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)
        if args.distilled_flag:
            data_path = "local_datasets/distilled_cifar/"+ args.distillation_mode +"/cifar10_ipc"+args.ipc
            distilled_traindataset = distilled_datasets.CustomDataset(data_path=data_path, transform=transform, ipc=args.ipc)
        from archs.transformer_distilled_pruning import cait, simplevit, swin, vit, vit_small, convnet
        num_classes = 10
        channel = 3

    elif args.dataset == "cifar100_distilled":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        if args.distilled_flag:
            data_path = "local_datasets/distilled_cifar/"+ args.distillation_mode +"/cifar100_ipc"+args.ipc
            distilled_traindataset = distilled_datasets.CustomDataset(data_path=data_path, transform=transform, ipc=args.ipc)
        from archs.transformer_distilled_pruning import cait, simplevit, swin, vit, vit_small
        num_classes = 100
        channel = 3


    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    if args.distilled_flag:
        distilled_train_loader = torch.utils.data.DataLoader(distilled_traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)
    elif args.arch_type == "cait":
        model = cait.CaiT(
            image_size = args.size,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05
        ).to(device)
    elif args.arch_type == "cait_small":
        model = cait.CaiT(
            image_size = 32,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05
        ).to(device)
    elif args.arch_type == "swin":
        model = swin.swin_t(
            window_size=args.patch_size,
            num_classes=num_classes,
            downscaling_factors=(2,2,2,1)
        ).to(device)
    elif args.arch_type == "simplevit":
        model = simplevit.SimpleViT(
            image_size = args.size,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,
            heads = 8,
            mlp_dim = 512
        ).to(device)
    elif args.arch_type == "vit":
        model = vit.ViT(
            image_size = args.size,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        ).to(device)
    elif args.arch_type == "vit_small":
        model = vit_small.ViT(
            image_size = args.size,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        ).to(device)
    elif args.arch_type == "vit_tiny":
        model = vit_small.ViT(
            image_size = args.size,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 4,
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        ).to(device)
    elif args.arch_type == "convnet":
        net_width, net_depth, net_act, net_norm, net_pooling = utils.get_default_convnet_setting()
        model = convnet.ConvNet(
            channel=channel, 
            num_classes=num_classes, 
            net_width=net_width,
            net_depth=net_depth, 
            net_act=net_act, 
            net_norm=net_norm, 
            net_pooling=net_pooling, 
            im_size = (32,32)
        ).to(device)


    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization
    if os.path.isfile(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{replication}/initial_state_dict_{args.prune_type}.pth.tar"):
        model = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{replication}/initial_state_dict_{args.prune_type}.pth.tar")
        initial_state_dict = copy.deepcopy(model.state_dict())
    else:
        if args.distilled_flag:
            sys.exit()
        model.apply(weight_init)

        # Copying and Saving Initial State
        initial_state_dict = copy.deepcopy(model.state_dict())
        utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{replication}/")
        torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{replication}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    time_taken = 0.0
    ITERATION = args.prune_iterations
    START, COMP = load_iterations(args, replication)
    if START > ITERATION or START < 0:
        print(f"Invalid argument(start_iter : {args.start_iter}, prune_iterations : {args.prune_iterations}, load_iterations : {START})")
        return 0
    elif START == ITERATION:
        print("This Iteration has already been completed")
        return 0
    elif START:
        comp, bestacc, time_taken, all_loss, all_accuracy = load_datas(args, replication, COMP)
        step = 0
        del COMP
    else:
        del COMP
        comp = np.zeros(ITERATION,float)
        bestacc = np.zeros(ITERATION,float)
        time_taken = np.zeros(ITERATION,float)
        step = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)


    for _ite in range(START, ITERATION):
        if (time.time() - start_time)/3600 > 23:
            print(f'time_limit')
            sys.exit()
        if not _ite == 0:
            prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                #if args.arch_type == "fc1":
                #    model = fc1.fc1().to(device)
                #elif args.arch_type == "lenet5":
                #    model = LeNet5.LeNet5().to(device)
                #elif args.arch_type == "alexnet":
                #    model = AlexNet.AlexNet().to(device)
                #elif args.arch_type == "vgg16":
                #    model = vgg.vgg16().to(device)  
                #elif args.arch_type == "resnet18":
                #    model = resnet.resnet18().to(device)   
                #elif args.arch_type == "densenet121":
                #    model = densenet.densenet121().to(device)   
                #else:
                #    print("\nWrong Model choice\n")
                #    exit()
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{replication}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{replication}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy
        end_time = time.time()
        time_taken[_ite]=round(end_time-start_time, 3)

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type},{replication})") 
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{replication}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_all_accuracy_{comp1}.dat")
        comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_compression_{comp1}.dat")
        bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_bestaccuracy_{comp1}.dat")
        time_taken.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_time_taken_{comp1}.dat')

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)
        
        write_iterations(args, replication, _ite, comp1) 


    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_bestaccuracy.dat")
    time_taken.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_time_taken.dat')

    for c in comp:
        if path.exists(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_compression_{c}.dat"):
            os.remove(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_compression_{c}.dat")
        if path.exists(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_bestaccuracy_{c}.dat"):
            os.remove(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_bestaccuracy_{c}.dat")
        if path.exists(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_time_taken_{c}.dat'):
            os.remove(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_time_taken_{c}.dat')
       

    # Plotting Best Accuracy
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type},{replication})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{replication}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()

    # Plotting Time
    plt.plot(a, time_taken, c="green", label="time_taken")
    plt.title(f'Time taken for each iteration to end ({args.dataset},{args.arch_type},{replication})')
    plt.xlabel("iterations")
    plt.ylabel("time (minutes)")
    plt.legend()
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{replication}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_TimeTaken.png", dpi=1200)
    plt.close()
   
# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                if p is None:
                    print(f'{name} : {p}')
                # else:
                #     print(f'{name}')
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def load_iterations(args, replication):
    if path.exists(f'{os.getcwd()}/logs/{args.arch_type}/{args.dataset}/{replication}/iteration.txt'):
        f = open(f'{os.getcwd()}/logs/{args.arch_type}/{args.dataset}/{replication}/iteration.txt', 'r')
        iter_num, comp = f.readline().split(' ')
        f.close()
        return int(iter_num) + 1, float(comp)
    else:
        return args.start_iter, 0.0
    
def write_iterations(args, replication, iter_num, comp):
    utils.checkdir(f"{os.getcwd()}/logs/{args.arch_type}/{args.dataset}/{replication}")
    f = open(f'{os.getcwd()}/logs/{args.arch_type}/{args.dataset}/{replication}/iteration.txt', 'w')
    data = f'{iter_num} {comp}'
    f.write(data)
    f.close()

def load_datas(args, replication, COMP):
    global mask
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/")
    comp = np.load(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_compression_{COMP}.dat", allow_pickle=True)
    bestacc = np.load(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_bestaccuracy_{COMP}.dat", allow_pickle=True)
    time_taken = np.load(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_time_taken_{COMP}.dat', allow_pickle=True)
    all_loss = np.load(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_all_loss_{COMP}.dat", allow_pickle=True)
    all_accuracy = np.load(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_all_accuracy_{COMP}.dat", allow_pickle=True)
    with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{replication}/{args.prune_type}_mask_{COMP}.pkl", 'rb') as fp:
        mask = pickle.load(fp)
    return comp, bestacc, time_taken, all_loss, all_accuracy

#def remove

if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")
    # Argument Parser for distilled pruning with transformer
    parser.add_argument("--size", default=32, type=int, help="Image size"), 
    parser.add_argument("--patch_size", default=4, type = int, help="Patch size")
    parser.add_argument("--dimhead", default=512, type = int)
    parser.add_argument("--replications", default=0, type=int, help="Total number of replications")
    parser.add_argument("--start_replication", default=0, type=int, help="Start number of replication")
    parser.add_argument("--distilled_flag", default=False, type=bool, help="If True else False")
    parser.add_argument("--distillation_mode", type=str, help="dataset distillation method, choice : dc | tm")
    parser.add_argument("--ipc", type=int, help="image per class\ndc mode choice : 1 | 10\ntm mode choice : 1 | 10 | 50")
    

    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    for i in range(args.start_replication, args.replications):
        print('-'*20 +f'replication : {i}'+ '-'*20)
        main(args, ITE=1, replication = i)
