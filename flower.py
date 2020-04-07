import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data import Subset
from PIL import Image
import json
import time,os
data_dir = 'data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])

data_transforms = {}

# transforms to train data set
data_transforms['train'] = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        ]),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        normalize_mean,
        normalize_std)
    ])

# transforms to valid data set
data_transforms['valid'] = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        normalize_mean,
        normalize_std)
    ])
# Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train_data'] = datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train'])
valid_dataset_to_split = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid'])
# obtain validation and training datasets that will be used to evaluate the network
valid_data_index_list = []
test_data_index_list = []
for index in range(0, len(valid_dataset_to_split), 2):
    valid_data_index_list.append(index)
    test_data_index_list.append(index+1)
image_datasets['valid_data'] = Subset(valid_dataset_to_split, valid_data_index_list)
image_datasets['test_data'] = Subset(valid_dataset_to_split, test_data_index_list)
# print(len(image_datasets['train_data']))
# print(len(image_datasets['valid_data']))
# print(len(image_datasets['test_data']))
# Using the image datasets and the transforms, define the dataloaders
dataloaders = {}
dataloaders['train_data'] = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=16, shuffle=True, num_workers=2)
dataloaders['valid_data'] = torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=16, shuffle=False, num_workers=2)
dataloaders['test_data'] = torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=16, shuffle=False, num_workers=2)

# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)
cat_to_name = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
class_to_idx = image_datasets['train_data'].class_to_idx
# print(class_to_idx)
cat_label_to_name = {}
for cat, label in class_to_idx.items():
    name = cat_to_name.get(cat)
    cat_label_to_name[label] = name
# print(cat_label_to_name)
# def imgview(img, title, ax):
#     # un-normalize
#     for i in range(img.shape[0]):
#         img[i] = img[i] * normalize_std[i] + normalize_mean[i]
#     # convert from Tensor image
#     ax.imshow(np.transpose(img, (1, 2, 0)))
#     ax.set_title(title)
# dataiter = iter(dataloaders['train_data'])
# images, labels = dataiter.next()
# images = images.numpy() # convert images to numpy for display
# # show some test images
# fig = plt.figure(figsize=(15, 15))
# fig_rows, fig_cols = 4, 4
# for index in np.arange(fig_rows*fig_cols):
#     img = images[index]
#     label = labels[index].item()
#     title = 'lable:{}\n{}'.format(label, cat_label_to_name[label])
#     ax = fig.add_subplot(fig_rows, fig_cols, index+1, xticks=[], yticks=[])
#     imgview(img, title, ax)
# plt.savefig('a.png')
# Freeze parameters so we don't backprop through them
def freeze_parameters(root, freeze=True):
    [param.requires_grad_(not freeze) for param in root.parameters()]
# Create a new classifier
def create_classifier(input_size, output_size, hidden_layers=[], dropout=0.5,
                      activation=nn.RReLU(), output_function=nn.LogSoftmax(dim=1)):
    dict = OrderedDict()
    if len(hidden_layers) == 0:
        dict['layer0'] = nn.Linear(input_size, output_size)
    else:
        dict['layer0'] = nn.Linear(input_size, hidden_layers[0])
        if activation:
            dict['activ0'] = activation
        if dropout:
            dict['drop_0'] = nn.Dropout(dropout)
        #for layer_in, layer_out in range(len(hidden_layers)):
        for layer, layer_in in enumerate(zip(hidden_layers[:-1],hidden_layers[1:])):
            dict['layer'+str(layer+1)] = nn.Linear(layer_in[0],layer_in[1])
            if activation:
                dict['activ'+str(layer+1)] = activation
            if dropout:
                dict['drop_'+str(layer+1)] = nn.Dropout(dropout)
        dict['output'] = nn.Linear(hidden_layers[-1], output_size)
    if output_function:
        dict['output_function'] = output_function
    return nn.Sequential(dict)
# Build and train your network
def create_network(model_name='resnet50', output_size=102, hidden_layers=[1000]):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
    # Replace the model classifier
    model.fc = create_classifier(num_ftrs, output_size, hidden_layers)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'resnet18'
output_size = 102
hidden_layers = [1000]
model = create_network(model_name, output_size, hidden_layers)
model.to(device)
def train(epochs, model, optimizers, lr_scheduler=None,
          dataloaders=dataloaders, state_dict=None,
          checkpoint_path="checkpoint.pt", accuracy_target=None,
          show_graphs=False):
              
    if state_dict == None:
        state_dict = {
            'elapsed_time': 0,
            'trace_log': [],
            'trace_train_loss': [],
            'trace_train_lr': [],
            'valid_loss_min': np.Inf,
            'trace_valid_loss': [],
            'trace_accuracy': [],
            'epochs_trained': 0}
        state_dict['trace_log'].append('PHASE ONE')
    for epoch in range(1,epochs+1):
            
            try:
                lr_scheduler.step() # if instance of _LRScheduler
            except TypeError:
                try:
                    if lr_scheduler.min_lrs[0] == lr_scheduler.optimizer.param_groups[0]['lr']:
                        break
                    lr_scheduler.step(valid_loss) # if instance of ReduceLROnPlateau
                except NameError: # valid_loss is not defined yet
                    lr_scheduler.step(np.Inf)
            except:
                pass # do nothing
    
            epoch_start_time = time.time()
            #####################
            #       TRAIN       #
            #####################
            train_loss = 0
            model.train()
            for images, labels in dataloaders['train_data']:
                # Move tensors to device
                images, labels = images.to(device), labels.to(device)
    
                # Clear optimizers
                [opt.zero_grad() for opt in optimizers]
    
                # Pass train batch through model feed-forward
                output = model(images)
    
                # Calculate loss for this train batch
                batch_loss = criterion(output, labels)
                # Do the backpropagation
                batch_loss.backward()
    
                # Optimize parameters
                [opt.step() for opt in optimizers]
    
                # Track train loss
                train_loss += batch_loss.item()*len(images)
    
            # Track how many epochs has already run
            state_dict['elapsed_time'] += time.time()-epoch_start_time
            state_dict['epochs_trained'] += 1
    
            #####################
            #      VALIDATE     #
            #####################
            valid_loss = 0
            accuracy = 0
            top_class_graph = []
            labels_graph = []
            # Set model to evaluation mode
            model.eval()
            with torch.no_grad():
                for images, labels in dataloaders['valid_data']:
                    labels_graph.extend( labels )
    
                    # Move tensors to device
                    images, labels = images.to(device), labels.to(device)
    
                    # Get predictions for this validation batch
                    output = model(images)
    
                    # Calculate loss for this validation batch
                    batch_loss = criterion(output, labels)
                    # Track validation loss
                    valid_loss += batch_loss.item()*len(images)
    
                    # Calculate accuracy
                    output = torch.exp(output)
                    top_ps, top_class = output.topk(1, dim=1)
                    top_class_graph.extend( top_class.view(-1).to('cpu').numpy() )
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()*len(images)
    
            #####################
            #     PRINT LOG     #
            #####################
            
            # calculate average losses
            train_loss = train_loss/len(dataloaders['train_data'].dataset)
            valid_loss = valid_loss/len(dataloaders['valid_data'].dataset)
            accuracy = accuracy/len(dataloaders['valid_data'].dataset)
    
            state_dict['trace_train_loss'].append(train_loss)
            try:
                state_dict['trace_train_lr'].append(lr_scheduler.get_lr()[0])
            except:
                state_dict['trace_train_lr'].append(
                    optimizers[0].state_dict()['param_groups'][0]['lr'])
            state_dict['trace_valid_loss'].append(valid_loss)
            state_dict['trace_accuracy'].append(accuracy)
    
            # print training/validation statistics 
            log = 'Epoch: {}: \
                   lr: {:.8f}\t\
                   Training Loss: {:.6f}\t\
                   Validation Loss: {:.6f}\t\
                   Validation accuracy: {:.2f}%\t\
                   Elapsed time: {:.2f}'.format(
                        state_dict['epochs_trained'],
                        state_dict['trace_train_lr'][-1],
                        train_loss,
                        valid_loss,
                        accuracy*100,
                        state_dict['elapsed_time']
                        )
            state_dict['trace_log'].append(log)
            print(log)
    
            # save model if validation loss has decreased
            if valid_loss <= state_dict['valid_loss_min']:
                print('Validation loss decreased: \
                      ({:.6f} --> {:.6f}).   Saving model ...'
                      .format(state_dict['valid_loss_min'],valid_loss))
    
                checkpoint = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizers[0].state_dict(),
                              'training_state_dict': state_dict}
                if lr_scheduler:
                    checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
                torch.save(checkpoint, checkpoint_path)
                state_dict['valid_loss_min'] = valid_loss
    
            if show_graphs:
                plt.figure(figsize=(25,8))
                plt.plot(np.array(labels_graph), 'k.')
                plt.plot(np.array(top_class_graph), 'r.')
                plt.show()
    
                plt.figure(figsize=(25,5))
                plt.subplot(1,2,1)
                plt.plot(np.array(state_dict['trace_train_loss']), 'b', label='train loss')
                plt.plot(np.array(state_dict['trace_valid_loss']), 'r', label='validation loss')
                plt.plot(np.array(state_dict['trace_accuracy']), 'g', label='accuracy')
    
                plt.subplot(1,2,2)
                plt.plot(np.array(state_dict['trace_train_lr']), 'b', label='train loss')
    
                plt.show()
    
    
            # stop training loop if accuracy_target has been reached
            if accuracy_target and state_dict['trace_accuracy'][-1] >= accuracy_target:
                break
    
    return state_dict

def test_model(dataloader=dataloaders['test_data'], show_graphs=True):
    #####################
    #       TEST        #
    #####################
    criterion = nn.NLLLoss()
    test_loss = 0
    accuracy = 0
    top_class_graph = []
    labels_graph = []
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels_graph.extend( labels )

            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Get predictions for this test batch
            output = model(images)

            # Calculate loss for this test batch
            batch_loss = criterion(output, labels)
            # Track validation loss
            test_loss += batch_loss.item()*len(images)

            # Calculate accuracy
            output = torch.exp(output)
            top_ps, top_class = output.topk(1, dim=1)
            top_class_graph.extend( top_class.view(-1).to('cpu').numpy() )
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

    #####################
    #     PRINT LOG     #
    #####################

    # calculate average losses
    test_loss = test_loss/len(dataloader.dataset)
    accuracy = accuracy/len(dataloader.dataset)

    # print training/validation statistics 
    log = f'Test Loss: {test_loss:.6f}\t\
           Test accuracy: {(accuracy*100):.2f}%'
    print(log)

    if show_graphs:
        plt.figure(figsize=(25,13))
        plt.plot(np.array(labels_graph), 'k.')
        plt.plot(np.array(top_class_graph), 'r.')
        plt.show()
        plt.savefig('a.png')
def load_model(checkpoint_path, state_dict):
    try:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['training_state_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        fc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass

    return state_dict

# Define a loss function
criterion = nn.NLLLoss() # Cross-Entropy (LogSoftmax and NLLLoss)

# Define how many times each phase will be running
PHASE_ONE = 100
PHASE_TWO = 20
PHASE_THREE = 10
TEST = True
# Define the phases
if PHASE_ONE > 0:
    freeze_parameters(model)
    freeze_parameters(model.fc, False)
    fc_optimizer = optim.Adagrad(model.fc.parameters(), lr=0.01, weight_decay=0.001)
    optimizers = [fc_optimizer]
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min',
                                                   factor=0.1, patience=5,
                                                   threshold=0.01, min_lr=0.00001)
    checkpoint_path = "checkpoint_phase_one.pt"
    state_dict = train(PHASE_ONE, model, optimizers, lr_scheduler=lr_scheduler,
                       state_dict=None, accuracy_target=None,
                       checkpoint_path=checkpoint_path)

    print(*state_dict['trace_log'], sep="\n")
    state_dict = load_model(checkpoint_path, state_dict)
if PHASE_TWO > 0:
    state_dict['trace_log'].append('PHASE TWO')
    freeze_parameters(model, False)
    conv_optimizer = optim.Adagrad(model.parameters(), lr=0.0001, weight_decay=0.001)
    optimizers = [fc_optimizer, conv_optimizer]
    checkpoint_path = "checkpoint_phase_two.pt"
    state_dict = train(PHASE_TWO, model, optimizers, lr_scheduler=None,
                       state_dict=state_dict, accuracy_target=None,
                       checkpoint_path=checkpoint_path)
    print(*state_dict['trace_log'], sep="\n")
    state_dict = load_model(checkpoint_path, state_dict)
if PHASE_THREE > 0:
    state_dict['trace_log'].append('PHASE THREE')

    freeze_parameters(model)
    freeze_parameters(model.fc, False)

    optimizers = [fc_optimizer]
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(fc_optimizer, milestones=[0], gamma=0.01)
        
    checkpoint_path = "checkpoint_phase_three.pt"
    
    state_dict = train(PHASE_THREE, model, optimizers, lr_scheduler=lr_scheduler,
                       state_dict=state_dict, accuracy_target=None,
                       checkpoint_path=checkpoint_path)

    print(*state_dict['trace_log'], sep="\n")

    state_dict = load_model(checkpoint_path, state_dict)

if TEST:
    test_model()

    
    



