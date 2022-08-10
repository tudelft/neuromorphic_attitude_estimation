import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm import tqdm, trange
from copy import deepcopy
import wandb
from pympler.tracker import SummaryTracker
# TODO: Loss function is now hard-coded, should be changed

def train(model, device, train_loader, optimizer, plot_results=False):
    model.train()
    losses = []
    torch.seed()
    
    with tqdm(total=min(15, len(train_loader))) as t:
        counter = 0
        for (data, target) in train_loader:

            # RNN:
            hidden = model.init_hidden(40)

            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            target = target.permute(1, 0, 2)

            optimizer.zero_grad()
            # ti = time()
            # output , _ = model(data)
            output , hidden = model(data, hidden)


            # print(f'calculating model output: {time() - ti}')
            loss = torch.nn.functional.mse_loss(output, target)
            # ti = time()
            loss.backward()

            # RNN:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # print(f'Performing backward step: {time() - ti}')
            # ti = time()
            optimizer.step()
            # print(f'Performing update step: {time() - ti}')
            losses.append(loss.item())
            
            if plot_results:
                # plt.clf()
                # plt.subplot(2, 1, 1)
                # plt.plot(output[:, 0, 0].detach().numpy())
                # plt.plot(target[:, 0, 0].detach().numpy())

                # plt.subplot(2, 1, 2)
                # plt.plot(output[:, 0, 1].detach().numpy())
                # plt.plot(target[:, 0, 1].detach().numpy())
                # # plt.plot(model.recording.lif0.z[:, 0, :].detach().numpy())
                # plt.pause(0.05)

                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(output[0, :, 0].detach().numpy())
                plt.plot(target[0, :, 0].detach().numpy())

                plt.subplot(2, 1, 2)
                plt.plot(output[0, :, 1].detach().numpy())
                plt.plot(target[0, :, 1].detach().numpy())
                # plt.plot(model.recording.lif0.z[:, 0, :].detach().numpy())
                plt.pause(0.05)

            # update the loading bar
            t.set_postfix(loss="{:05.4f}".format(loss.item()))
            # time_data = np.linspace(0, len(output[:, 0, 0]) / 100, len(output[:, 0, 0]))
            # pitch = [output[:, 0, 0].detach().numpy(), target[:, 0, 0].detach().numpy()]
            # roll = [output[:, 0, 1].detach().numpy(), target[:, 0, 1].detach().numpy()]

            # pitch_plot = wandb.plot.line_series(time_data, pitch, title="Pitch")
            # roll_plot = wandb.plot.line_series(time_data, roll, title="Roll")
            # wandb.log({"loss": loss.item(),
            #             "pitch_plot" : pitch_plot,
            #             "roll_plot" : roll_plot})
            # del pitch_plot, roll_plot
            wandb.log({"loss": loss.item()})
            t.update()
            counter += 1
            if counter == 15:
                break
        mean_loss = np.mean(losses)
        t.set_postfix(mean_loss="{:05.4f}".format(mean_loss))
        t.update()
    return losses, mean_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target = target.permute(1, 0, 2)

            # RNN:
            hidden = model.init_hidden(40)
            output, hidden = model(data, hidden)

            # output, _ = model(data)
            test_loss += torch.nn.functional.mse_loss(
                output, target
            ).item()
            break

    # test_loss /= len(test_loader.dataset)
    return test_loss

def fit(model, train_loader, val_loader, optimizer, EPOCHS, out_dir, device):
    # training_losses = []
    mean_losses = []
    test_losses = []
    time_hist = []
    best_model = None
    best_loss = np.inf
    # tracker = SummaryTracker()
    torch.autograd.set_detect_anomaly(True)

    try:
        for epoch in trange(EPOCHS):
            epoch_t = time()
            training_loss, mean_loss = train(model, device, train_loader, optimizer)
            # print('after train')
            # tracker.print_diff()
            test_loss = test(model, device, val_loader)
            wandb.log({"val_loss": float(test_loss)})
            # print('after test')
            # tracker.print_diff()
            # training_losses += training_loss
            mean_losses.append(float(mean_loss))
            test_losses.append(float(test_loss))
            time_hist.append(time() - epoch_t)
            
            if test_loss < best_loss:
                best_model = deepcopy(model.state_dict())
                model_save_file = f'{out_dir}/model.pt'
                with open(model_save_file, 'wb') as f:
                    torch.save(best_model, f)
                best_loss = mean_loss
            # tracker.print_diff()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        
    plt.show()
    print(f"Best test loss: {float(np.min(test_losses))}")
    # best_model = best_model.state_dict()
    best_fitness = min(mean_losses)
    avg_time = float(np.mean(time_hist))
    return best_model, best_fitness, epoch, avg_time, mean_losses, test_losses