from data_loader import *
import argparse
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from networks import *
print(['Using device: ', torch.cuda.get_device_name(0)])

def achieve_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--save_dir', type=str, default='results/',
            help='Path to save the trained models (default=results/).')

    parse.add_argument('--image_path', type=str, default='C:\\Users\\cherb\\Documents\\Github\\pytorch_tutorials\\2d_projected_data_right',
            help='Path to image folder.'
                 'image size = 240, 320')

    parse.add_argument('--label_path', type=str, default='C:\\Users\\cherb\\Documents\\Github\\pytorch_tutorials\\ages_scores.pkl',
            help='Path to labels file.')

    parse.add_argument('--net', type=str, default='SimpleResNetDLTK2',
            help='Network architecture (default=SimpleResNetDLTK2).'
                 'options: AlexNet, SimpleResNet, '
                 'SimpleResNetDLTK, SimpleResNetDLTK2')

    parse.add_argument('--save_every', type=int, default=1,
                        help='After how many iter to save the model.')

    parse.add_argument('--epochs', type=int, default=100,
            help='Num epochs (default=200).')

    parse.add_argument('--batch_size', type=int, default=32,
            help='train batch_size (default=32).')

    parse.add_argument('--val_batch_size', type=int, default=32,
            help='val batch_size.')

    parse.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default=0.001)')

    parse.add_argument('--betas', type=float, default=(0.5, 0.999),
                        help='betas for Adam optim (default=0.9, 0.999)')

    parse.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default=0.9)')

    parse.add_argument('--loss', type=str, default='l1',
                        help='regression loss function (default=l1).'
                             'Options: l1, l2')

    parse.add_argument('--loss_regularisation', type=float, default=0,
            help='loss (l2) regularisation, weight_decay (default=0.0001).')

    parse.add_argument('--optimizer', type=str, default='SGD',
            help='optimizers (default=SGD).'
                 'options: Adagrad, Adam, RMSprop, SGD')

    parse.add_argument('--aug', type=bool, default=True,
                        help='whether to augment data')

    parse.add_argument('--aug_gauss', type=float, default=(0, 0.05),
            help='Gaussian noise augmentation (default=(0, 0.05)). '
                 'Mean and STD of the gaussian')

    parse.add_argument('--aug_flip_lr', type=float, default=0.0,
            help='flip augmentation (default=0.5). '
                    'probability of flipping left right')

    parse.add_argument('--aug_elastic', type=float, default=((0, 70), (4, 6)),
            help='elastic deformation augmentation (default=((0, 70), (4, 6))).'
                 'Strength of the displacement: higher values mean that pixels are moved further'
                 'Smoothness of the displacement: higher values lead to smoother patterns')

    args = parse.parse_args()
    return args

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def line_best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b


def train_network_2d_data(args_in=None):

    args = vars(achieve_args())

    if args_in:
        args.update(args_in)

    # Setting parameters
    timestr = time.strftime("%d%m%Y-%H%M")
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    args['time_date'] = timestr
    experiment = args['net'] + '_'
    directory = experiment + timestr

    path = os.path.join(__location__, args['save_dir'], directory)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/parameters.json', 'w') as file:
        json.dump(args, file, indent=4, sort_keys=True)

    args['experiment_name'] = directory

    args['random_seed'] = 8

    dirpath = os.getcwd()

    if args['aug']:
        transforms = torchvision.transforms.Compose([ImgAugTransform(gauss_noise=[args['aug_gauss'][0], args['aug_gauss'][1]],
                                                                 flip_lr=args['aug_flip_lr'],
                                                                 elastic=[args['aug_elastic'][0], args['aug_elastic'][1]])])
    else:
        transforms = None

    dataset = Dataset2D(root_dir=dirpath, image_path=args['image_path'],
                        label_path=args['label_path'], transform= transforms)

    train, val = train_valid_split(dataset, split_fold=10, random_seed=args['random_seed'])


    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args['batch_size'], shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=args['val_batch_size'], shuffle = True)

    train_iter = iter(train_dataloader)
    image, _ = train_iter.next()
    args['image_size'] = [image.size(1), image.size(2), image.size(3)]

    args['cuda'] = torch.cuda.is_available()
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    if args['net'] == 'AlexNet':
        net = AlexNet()
    elif args['net'] == 'SimpleResNet':
        net = SimpleResNet(PreActBlock)
    elif args['net'] == 'SimpleResNetDLTK':
        net = SimpleResNetDLTK(PreActBlock)
    elif args['net'] == 'SimpleResNetDLTK2':
        net = SimpleResNetDLTK2(PreActBlock)

    net.apply(weights_init)
    net = net.to(device)

    if args['loss'] == 'l1':
        criterion = nn.L1Loss()  # L1Loss (mean absolute error)
    elif args['loss'] == 'l2':
        criterion = nn.MSELoss()  # MSEloss (l2 loss)

    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['loss_regularisation'], betas=args['betas'])
    elif args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['loss_regularisation'])
    elif args['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['loss_regularisation'])
    elif args['optimizer'] == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], momentum=args['momentum'], weight_decay=args['loss_regularisation'])

    num_saves = (len(train_dataloader) // args['save_every'] +1)*args['epochs']
    all_error = np.zeros(num_saves)
    num_it_per_epoch = (len(train_dataloader) // (args['save_every']))
    epochs = np.arange(1, all_error.size + 1) / num_it_per_epoch
    all_val_error = np.zeros((len(val_dataloader)+1)*args['epochs'])
    epochs_val = np.arange(1, all_val_error.size + 1) / len(val_dataloader)
    mean_val_error = np.zeros(args['epochs'])

    if criterion:
        criterion = criterion.to(device)

    a = 0
    b = 0
    t0 = time.time()
    for epoch in range(args['epochs']):
        net.train()
        for i, (data, label) in enumerate(train_dataloader):

            data = data.to(device)
            label = label.to(device)

            x_out = net(data)
            x_out = x_out.squeeze(1)
            err = criterion(x_out, label)

            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            time_elapsed = time.time() - t0

            if ((i) % args['save_every'] == 0):
                print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Loss: {:.4f}'
                      .format(epoch, args['epochs'], i, len(train_dataloader), time_elapsed // 60, time_elapsed % 60,
                              err.item()))

                error = err.item()
                all_error[a] = error
                a = a + 1

                plt.figure()
                plt.plot(epochs[:a], all_error[:a], label='training loss')
                plt.xlabel('epochs')
                plt.legend()
                plt.title('Loss')
                plt.savefig(path + '/train_loss.png')
                plt.close()

        labels = np.zeros(0)
        predictions = np.zeros(0)
        val_error = np.zeros(len(val_dataloader))

        for i, (data, label) in enumerate(val_dataloader):

            data = data.to(device)
            label = label.to(device)
            x_out = net(data)
            x_out = x_out.squeeze(1)
            err = criterion(x_out, label)

            predictions = np.append(predictions, x_out.cpu().detach().numpy())
            labels = np.append(labels, label.cpu().detach().numpy())
            val_error[i] = err.item()
            all_val_error[b] = err.item()

            b = b + 1

        mean_val_error[epoch] = np.mean(val_error)
        print('Val Loss: {:.4f}'.format(mean_val_error[epoch]))

        if mean_val_error[epoch] >= np.min(mean_val_error[:epoch + 1]):
            args['train_error'] = np.min(all_error[:a])
            args['val_error'] = np.min(mean_val_error[:epoch+1])
            args['val_epoch'] = epoch

            with open(path + '/parameters.json', 'w') as file:
                json.dump(args, file, indent=4, sort_keys=True)

            torch.save(net.state_dict(), '%s/best_model.pt' % (path))

            x, y = line_best_fit(labels, predictions)
            yfit = [x + y * xi for xi in labels]
            plt.figure()
            plt.plot(labels, predictions, '+')
            plt.plot(labels, yfit, 'k', linewidth=1)
            plt.xlabel('true values')
            plt.ylabel('predicted values')
            plt.xlim([30, 45])
            plt.ylim([30, 45])
            plt.title('True vs predicted values plot')
            plt.savefig(path + '/val_accuracy_plot.png')
            plt.close()

        plt.figure()
        plt.plot(epochs_val[:b], all_val_error[:b], label='val loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.title('Loss')
        plt.savefig(path + '/val_loss.png')
        plt.close()

        np.save(path + '/train_error.npy', all_error)
        np.save(path + '/val_error.npy', mean_val_error)

    torch.save(net.state_dict(), '%s/last_model.pt' % (path))

    return args


if __name__ == '__main__':
    args = train_network_2d_data()
    print('finished training')
