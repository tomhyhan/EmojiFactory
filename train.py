import torch

def train_model(
    model : torch.nn.Module,
    train_set,
    validation_set,
    num_epochs,
    batch_size,
    loss_func,
    lr=1e-5,
    warmup_iterations=200,
    warmup_lr=1e-6,
    device="cpu"
):
    if warmup_iterations:
        optimizer = torch.optim.Adam(model.parameters(), lr=warmup_lr, betas=(0.9, 0.995))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.995))

    model.to(device)
    
    iterations = 0 
    for epoch in range(num_epochs):
        loss_history = []
        model.train()
        for batch in train_set:
            ques, ques_pos, ans, ans_pos, sentiments = batch
            
            ques = ques.to(device)
            ques_pos = ques_pos.to(device)
            ans = ans.to(device)
            ans_pos = ans_pos.to(device)
            sentiments = sentiments.to(device)
            
            gnd = ans[:,1:].contiguous().view(-1).long()
            
            if warmup_iterations and iterations == warmup_iterations:
                print(f"Warm up cycle ended, changing to normal learning rate: {lr:.4f}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            optimizer.zero_grad()
            pred = model(ques.long(), ques_pos, ans.long(), ans_pos, sentiments)
            loss = loss_func(pred, gnd)
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss)
            iterations += 1
            
        avg_loss = sum(loss_history) / len(loss_history)
        print(f"Epoch: {epoch + 1}, Iterations: {iterations}, Average loss: {avg_loss}")
    