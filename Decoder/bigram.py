
f = open('input.txt','r')
content = f.read()


vocabulary = list(set(content))
n = len(vocabulary)
stoi = {value:key for key,value in enumerate(vocabulary)}
itos = {key:value for key,value in enumerate(vocabulary)}
encode = lambda x : [stoi[i] for i in x]
decode = lambda x : "".join([itos[index] for index in x])


# Global Variables
batch_size = 4
block_size = 16
e_embd = 32
vocab_size = len(vocabulary)


# Training and validation data split 
# Train : val : test ==> 80 : 10 : 10
train_records = int(len(content)*0.8)
val_records = train_records + int(len(content)*0.1)
train_data = content[:train_records]
val_data = content[train_records:val_records]
test_data = content[val_records:]
print(len(train_data),len(val_data),len(test_data))


import torch
def batch(split):
    data = train_data if split == 'train' else val_data
    indexes = torch.randint(0,len(data) - block_size,(batch_size,))
    x_train = torch.stack([torch.tensor(encode(data[index:index+block_size])) for index in indexes])
    y_train = torch.stack([torch.tensor(encode(data[index+1:index+block_size+1])) for index in indexes])
    return x_train,y_train


from torch import nn
import torch.nn.functional as F
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self,x,y=None):
        logits = self.embedding(x)
        if y == None:
            return logits
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T,C),y.view(-1))
            return logits , loss

bigram = BigramLanguageModel()
x_train,y_train = batch('train')
logits,loss = bigram(x_train,y_train)

print()


@torch.no_grad()
def validate(model):
    average_over = 20
    train_loss = 0
    val_loss = 0
    for i in range(average_over):
        x_train,y_train = batch('train')
        x_val,y_val = batch('val')
        train_loss += model(x_train,y_train)[1].item()
        val_loss += model(x_val,y_val)[1].item()
    train_loss/=average_over
    val_loss/=average_over
    print(f"Training Loss : {train_loss} | Validation Loss : {val_loss}")
    


import torch.optim as optim
import matplotlib.pyplot as plt
%matplotlib inline

epochs = 10000
l_r = 1e-3
loss_values = []
model = BigramLanguageModel()
optimizer = optim.AdamW(model.parameters(),l_r)
for i in range(epochs):
    x_train,y_train = batch('train')
    logits,loss = model(x_train,y_train)
    loss_values.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        validate(model)

validate(model)



@torch.no_grad()
def generate(model,max_output_tokens,token):
    idx = token
    for i in range(max_output_tokens):
        logits = model(idx)
        logits = F.softmax(logits[:,-1,:],dim=-1)
        out = torch.multinomial(logits,num_samples=1)
        idx = torch.cat((idx,out),dim=1)
    return decode(idx[0].tolist())
print(generate(model,1000,torch.zeros(1,1).int()))







