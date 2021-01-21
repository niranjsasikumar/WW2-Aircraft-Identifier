import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

###################
##  Build Model  ##
###################

model = models.densenet161(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier_input = model.classifier.in_features
num_labels = 7

classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

device = torch.device("cpu")
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())


############################
##  Obtain Training Data  ##
############################

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_set = datasets.ImageFolder("./training_set", transform = transformations)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)


###################
##  Train Model  ##
###################

epochs = 20
progress = 0
for epoch in range(epochs):
    model.train()
    
    for inputs, labels in training_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
                    
    progress += 5
    print('Training: ' + str(progress) + '% complete.')


##################
##  Test Model  ##
##################
    
model.eval()

def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255
    
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    img = img[np.newaxis,:]
    
    image = torch.from_numpy(img)
    image = image.float()
    
    return image

def predict(image, model):
    output = model.forward(image)
    output = torch.exp(output)
    probs, classes = output.topk(1, dim=1)
    
    return probs.item(), classes.item()
    
aircraft = {0: 'A6M Zero',
            1: 'B-17 Flying Fortress',
            2: 'Bf 109',
            3: 'Il-2 Sturmovik',
            4: 'Ju 87 Stuka',
            5: 'P-51 Mustang',
            6: 'Spitfire',}

test_images = ['p-51.jpg', 'b-17.jpg', 'spitfire.jpg', 'il-2.jpg', 'bf109.jpg', 'ju87.jpg', 'a6m.jpg']

for image in test_images:
    image = process_image('./validation_set/' + image)
    top_prob, top_class = predict(image, model)
    
    print('We are ' + str(round(top_prob*100, 1)) + '% certain that this aircraft is a ' + aircraft[top_class] + '.')