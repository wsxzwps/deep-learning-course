import pickle
from solver import *
from softmax import *

with open('mnist.pkl', 'rb') as f:
    train_set, val_set, test_set = pickle.load(f,encoding='latin1')

train_data = np.array(train_set[0])
train_label = np.array(train_set[1])

val_data = np.array(val_set[0])
val_label = np.array(val_set[1])

test_data = np.array(test_set[0])
test_label = np.array(test_set[1])

data = {
    'X_train': train_data,
    'y_train': train_label,
    'X_val': val_data,
    'y_val': val_label
}
# model = LogisticClassifier(input_dim=20, hidden_dim=10, weight_scale=0.1, reg=0.01)


model = SoftmaxClassifier(hidden_dim=500, weight_scale=0.1, reg=0.01)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                'learning_rate': 0.3,
                },
                num_epochs=50, batch_size=100,
                print_every=100)
solver.train()
test_score = model.loss(test_data)
acc = 0
total = test_label.shape[0]
pred = np.argmax(test_score, axis=1)
acc = np.mean(pred==test_label)
print(acc)

val_score = model.loss(val_data)
acc = 0
total = val_label.shape[0]
pred = np.argmax(val_score, axis=1)
acc = np.mean(pred==val_label)
print(acc)
