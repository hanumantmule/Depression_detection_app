X_train = []
Y_train = []
X_test = []
Y_test = []

counter = 0

cut = 10
debt = 0

for i in range(len(text_features_train)):

    if text_ctargets_train[i] >= 10:
        times = 3 + debt if counter < 46 else 2 + debt
        #         print(times, text_features_train[i].shape, debt)
        for j in range(times):
            if (j + 1) * cut > len(text_features_train[i]):
                debt += 1
                continue
            X_train.append(text_features_train[i][j * cut:(j + 1) * cut])
            Y_train.append(text_ctargets_train[i])
            if debt > 0:
                debt -= 1
            counter += 1
    else:
        X_train.append(text_features_train[i][:cut])
        Y_train.append(text_ctargets_train[i])

for i in range(len(text_features_test)):
    X_test.append(text_features_test[i][:cut])
    Y_test.append(text_ctargets_test[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

config = {
    'num_classes': 2,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 1024,
    'batch_size': 8,
    'epochs': 200,
    'learning_rate': 5e-4,
    'hidden_dims': 128,
    'bidirectional': True
}

model = BiLSTM(config)


optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

criterion = nn.SmoothL1Loss()
max_f1 = -1
max_acc = -1
train_acc = -1
min_mae = 100


def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


def standard_confusion_matrix(y_test, y_test_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


def model_performance(y_test, y_test_pred_proba):

    y_test_pred = y_test_pred_proba.data.max(1, keepdim=True)[1]
    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred, conf_matrix


def plot_roc_curve(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(prefix + 'images/BiLSTM_roc.png')
    plt.close()


def train(epoch):
    global lr, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    for i in range(0, X_train.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_train.shape[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i + config['batch_size'])], Y_train[i:(i + config['batch_size'])]
        if False:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(
                torch.from_numpy(y)).cuda()
        else:
            #             x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y))
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(
                torch.from_numpy(y)).type(torch.FloatTensor)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.flatten(), y)
        loss.backward()
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

    train_acc = total_loss / batch_idx
    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\t Loss: {:.6f}\t '.format(
        epoch + 1, config['learning_rate'], total_loss / batch_idx))


def evaluate(model):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global max_f1, max_acc, min_mae
    with torch.no_grad():
        x, y = Variable(torch.from_numpy(X_test).type(torch.FloatTensor), requires_grad=True), Variable(
            torch.from_numpy(Y_test)).type(torch.FloatTensor)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.flatten(), y)
        total_loss += loss.item()
        #print(y, output)

        print('\nTest set: Average loss: {:.4f} \t MAE: {:.4f}\n'.format(total_loss, F.l1_loss(output.flatten(), y)))

        # custom evaluation metrics
        print('Calculating additional test metrics...')
        accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
        precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
        recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1-Score: {}\n".format(f1_score))
        print('=' * 89)

        if max_f1 <= f1_score and train_acc > 151:
            max_f1 = f1_score
            max_acc = accuracy
            save(model, 'BiLSTM_elmo_{}_{:.2f}'.format(config['hidden_dims'], max_f1))
            print('*' * 64)
            print('model saved: f1: {}\tacc: {}'.format(max_f1, max_acc))
            print('*' * 64)

    return total_loss


for ep in range(1, config['epochs']):
    train(ep)
    tloss = evaluate(model)

lstm_model = torch.load('E:\\Studies\\sem II\\WSC\\project\\final-app\\bilstm-text\\output\\bilstm.h5')
model = BiLSTM(config)
model.load_state_dict(lstm_model.state_dict())
evaluate(model)