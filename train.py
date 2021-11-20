import os
import glob
import argparse

import torch
import torch.optim as optim
import torchvision as tv
import numpy as np

import tqdm
from sklearn.metrics import confusion_matrix

from resnet import resnet18, resnet50

try:
    from torch.cuda import amp
    amp_train = True
except:
    amp_train = False


class_dict = {"good":0, "bad" :1}
#class_dict = {"good":0, "bad" :1, "outlier":2}

class Model(torch.nn.Module):
    def __init__(self, classes, base_model):
        super(Model, self).__init__()
        self.nc = classes
        self.base_model = base_model
        
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=64, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = torch.nn.Conv2d(in_channels=64,out_channels=128, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2   = torch.nn.BatchNorm2d(128)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv3 = torch.nn.Conv2d(in_channels=128,out_channels=256, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3   = torch.nn.BatchNorm2d(256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv4 = torch.nn.Conv2d(in_channels=256,out_channels=512, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn4   = torch.nn.BatchNorm2d(512)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv1x1 = torch.nn.Conv2d(in_channels=512,out_channels=1, 
                kernel_size=(1,1), stride=(1,1), padding=0)
        
        self.relu  = torch.nn.ReLU(inplace=True) 
        
        self.FC    = torch.nn.Linear(in_features=1024, out_features=self.nc)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp):
        inp = inp.unsqueeze(0)
        inp = inp.reshape(-1,3,16,16)
        #out = self.base_model(inp)
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv1x1(x)
       
        ##method 1 - Use FC layer to map 1024 to nc classes
        #return self.FC(x.view(-1)).unsqueeze(0)

        ## method 2 - take the mean of 1024 values and use logistic classifier 
        return torch.sigmoid(torch.mean(x))
         
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]

def mktrainval(args):
  """Returns train and validation datasets."""
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((512, 512)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(), 
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((512, 512)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  
  #folder_path = os.path.join(args.datadir, "train")
  #files  = sorted(glob.glob("%s/*/*.*" % folder_path))
  #labels = [int(file.split("/")[-2]) for file in files]
  #labels = [class_dict[file.split("/")[-2]] for file in files]
  #train_set = tv.datasets.ImageFolder(files, labels, train_tx)

  #folder_path = os.path.join(args.datadir, "val")
  #files  = sorted(glob.glob("%s/*/*.*" % folder_path))
  #labels = [int(file.split("/")[-2]) for file in files]
  #labels = [class_dict[file.split("/")[-2]] for file in files]
  #valid_set = tv.datasets.ImageFolder(files, labels, val_tx)

  train_set = tv.datasets.ImageFolder(os.path.join(args.datadir, "train"), train_tx)
  valid_set = tv.datasets.ImageFolder(os.path.join(args.datadir, "val"), val_tx)
  
  micro_batch_size = args.batch_size 
  micro_batch_size_val = micro_batch_size

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size_val, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader

def run_eval(model, data_loader, device, epoch, num_classes, criterion=None):
  
  print("Running validation...")
  
  # switch to evaluate mode
  model.eval()
  
  running_loss = 0
  running_corrects = 0

  preds = []
  gts   = []
  
  pbar = enumerate(data_loader)
  pbar = tqdm.tqdm(pbar, total=len(data_loader))

  #for b, (path, x, y) in pbar:
  for b, (inputs, labels) in pbar:
    with torch.no_grad():
      inputs = inputs.to(device)
      labels = labels.to(device)

      # compute output, measure accuracy and record loss.
      outputs = model(inputs)
      if criterion:
          #loss = criterion(output, labels).item()
          loss = criterion(outputs, labels[0].float()).item()
      else:
          loss = 0.0

      #_, preds_ = torch.max(output, 1)
      preds = outputs.item() > 0.5
    
      #preds.extend(preds_.cpu().numpy())
      #gts.extend(labels.cpu().numpy())
      
      # statistics
      running_loss += loss * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)

  #preds = np.array(preds)
  #gts   = np.array(gts)

  #print("Cij  is equal to the number of observations known to be in group i and predicted to be in group j")
  #print(confusion_matrix(gts, preds))

  eval_loss = running_loss / len(data_loader.dataset)
  eval_accuracy = running_corrects / float(len(data_loader.dataset))

  return eval_loss, eval_accuracy

def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b

def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def main(args):

  cp, cn = smooth_BCE(eps=0.1)

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Going to train on {device}")

  classes = args.classes

  train_set, valid_set, train_loader, valid_loader = mktrainval(args)
  
  model = Model(classes, resnet50(pretrained=True))
  model = torch.nn.DataParallel(model)
  model = model.to(device)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  start_epoch = 0


  if args.optimizer == "sgd":            
      # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
      if args.wd:
          optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
      else:
          optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optimizer == "adam":
      if args.wd:
          optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
      else:
          optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
  
  if args.scheduler == 'one_cycle':
      scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lrf, steps_per_epoch=1, epochs=args.num_epochs)
  elif args.scheduler == "step_lr":
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
              milestones=[20,40,60],
              gamma=0.1,
              last_epoch=-1)

  # Resume fine-tuning if we find a saved model.
  if not os.path.exists(args.logdir):
      os.makedirs(args.logdir)
  savename = os.path.join(args.logdir, "best.pth.tar")
  
  if args.weights and os.path.exists(args.weights):
      model.load_state_dict(torch.load(args.weights)['model'])
  if args.evaluate:
      val_loss, val_acc = run_eval(model, valid_loader, device, -1, classes)
      return
  
  best_acc = -1
  log_file = open(os.path.join(args.logdir, "training_log.txt"), "w")
  
  #criterion = torch.nn.CrossEntropyLoss().to(device)
  criterion = torch.nn.BCELoss().to(device)
  
  print("Starting training!")
  for epoch in range(start_epoch, args.epochs):
        
      model.train() 
      running_loss = 0
      running_corrects = 0

      pbar = enumerate(train_loader)
      pbar = tqdm.tqdm(pbar, total=len(train_loader))

      all_top1, all_top5 = [], []
      for param_group in optimizer.param_groups:
          lr = param_group["lr"]
          
      #for batch_id, (path, x, y) in pbar:
      for batch_id, (inputs, labels) in pbar:

          # Schedule sending to GPU(s)
          inputs = inputs.to(device)
          labels = labels.to(device)
          
          # zero the parameter gradients
          optimizer.zero_grad()

          # compute output
          outputs = model(inputs)

          #loss = criterion(outputs, labels)
          loss = criterion(outputs, labels[0].float())
          loss.backward()
          
          # Update params
          optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)
          preds = outputs.item() > 0.5
          #_, preds = torch.max(outputs, 1)
          running_corrects += torch.sum(preds == labels.data)

      train_loss = running_loss / len(train_loader.dataset)
      train_accuracy = running_corrects / float(len(train_loader.dataset))
      
      # Run evaluation and save the model.
      val_loss, val_acc = run_eval(model, valid_loader, device, epoch, classes, criterion)

      best = val_acc > best_acc
      if best:
          best_acc = val_acc
          torch.save({
              "epoch": epoch,
              "val_loss": val_loss,
              "val_acc": val_acc,
              "train_acc": train_accuracy,
              "model": model.state_dict(),
              "optim" : optimizer.state_dict(),
          }, savename)
      s = "Epoch: {:03d} LR: {:.8f} Train Loss: {:.5f} Train Acc: {:.3f} Eval Loss: {:.5f} Eval Acc: {:.3f}" \
                              .format(epoch + 1, lr, train_loss, train_accuracy, val_loss, val_acc)
      print(s)
      if scheduler is not None:
          scheduler.step()
      else:
          for x in optimizer.param_groups:
              x['lr'] = lf(epoch+1)

      log_file.write(s + "\n")
  log_file.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='matdun cv assignment')
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--logdir", required=True,
                      help="Path where logs will be written")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--evaluate", action="store_true")
  parser.add_argument("--weights", type=str, required=False)
  parser.add_argument("--epochs", type=int, required=True)
  parser.add_argument("--batch_size", type=int, required=True)
  parser.add_argument("--classes", type=int, required=True)
  parser.add_argument('--noise', type=float, default=0.02, help='range is 0.0-1.0, The measure of noise that can present in the data')
  parser.add_argument("--optimizer", type=str, default = "", choices=["sgd", "adam"])
  parser.add_argument("--wd", action='store_true',
          help="Enable if you want to use weight decay during training")
  parser.add_argument('--scheduler', type=str,
          choices = ['one_cycle', 'lr_test', 'step_lr'],
          help='Tell which schedulers to use')
  parser.add_argument('--lr', type=float, default=0.01, 
          help='The starting learning rate')
  main(parser.parse_args())
