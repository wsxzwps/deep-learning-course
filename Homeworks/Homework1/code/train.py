from solver import *
from logistic import *
import pickle


with open('data.pkl', 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

train_data = np.array([raw_data[0][i] for i in range(500)])
train_label = np.array([raw_data[1][i] for i in range(500)])

val_data = np.array([raw_data[0][i] for i in range(500, 750)])
val_label = np.array([raw_data[1][i] for i in range(500, 750)])

test_data = np.array([raw_data[0][i] for i in range(750, 1000)])
test_label = np.array([raw_data[1][i] for i in range(750, 1000)])

data = {
    'X_train': train_data,
    'y_train': train_label,
    'X_val': val_data,
    'y_val': val_label
}
model = LogisticClassifier(input_dim=20, hidden_dim=22, weight_scale=0.1, reg=0.01)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                'learning_rate': 1,
                },
                num_epochs=200, batch_size=100,
                print_every=100)
solver.train()
test_score = model.loss(test_data)
acc = 0
mask = (test_score>0)
result = (mask == test_label)
acc = sum(result)/250
print(acc)
