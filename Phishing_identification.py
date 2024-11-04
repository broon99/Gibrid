import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from torch.utils.data import DataLoader, TensorDataset

# Глобальная переменная для скейлера
scaler = StandardScaler()

# Функция для проверки введенного URL
def check_url(hybrid_model, url, ip):
    # Извлекаем признаки из URL и IP
    length_url, num_special_chars, has_https, ip_feature = extract_url_features(url, ip)

    # Создаем массив признаков
    features = np.array([[length_url, num_special_chars, has_https, ip_feature]], dtype=np.float32)

    # Нормализуем данные

    features_scaled = scaler.fit_transform(features)

    # Преобразуем данные в формат PyTorch
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    # Получаем предсказание от гибридной модели
    hybrid_model.eval()
    with torch.no_grad():
        bayesian_output = hybrid_model(features_tensor)

    # Преобразуем вывод в вероятность
    probability = torch.sigmoid(bayesian_output).item()

    # Определяем, является ли URL фишинговым
    if probability > 0.75:
        print(f'URL: {url} - Вероятность фишинга: {probability:.2f} (Фишинговый)')
    else:
        print(f'URL: {url} - Вероятность фишинга: {probability:.2f} (Безопасный)')

# Функция для построения ROC-кривой
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Находим оптимальную точку
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC-кривая (площадь = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)  # Линия случайного выбора
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', label='Оптимальная точка (Threshold = {:.2f})'.format(optimal_threshold))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложноположительных результатов', fontsize=14)
    plt.ylabel('Доля истинноположительных результатов', fontsize=14)
    plt.title('ROC-кривая', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


# Функция для теста гибридной модели
def test_hybrid_model(hybrid_model, test_loader):
    hybrid_model.eval()
    correct_test = []
    y_test_list = []
    y_scores = []  # Список для хранения предсказанных вероятностей
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Получаем предсказания от RNN
            rnn_outputs = hybrid_model.rnn_model(X_batch)
            # Получаем вывод от байесовской сети
            bayesian_output = hybrid_model.bayesian_infer(rnn_outputs)

            # Сохраняем вероятности
            y_scores.extend(bayesian_output.cpu().numpy())
            # Изменяем размер y_batch до (batch_size,)
            y_batch = y_batch.view(-1)
            # Получаем предсказанные метки, пороговым значением Bayesian вывода
            predicted = (bayesian_output > 0.5).float().squeeze()
            # Сравниваем предсказанные метки с истинными метками и храним результаты
            correct_test.extend((predicted == y_batch).cpu().numpy())
            # Храним истинные метки для каждого пакета
            y_test_list.extend(y_batch.cpu().numpy())

    # Рассчитываем точность и полноту с помощью хранимых истинных меток и предсказанных меток
    accuracy = accuracy_score(y_test_list, correct_test)
    recall = recall_score(y_test_list, correct_test)
    precision = precision_score(y_test_list, correct_test)
    f1 = f1_score(y_test_list, correct_test)
    print(f'Общая точность гибридной модели на тестовой выборке: {accuracy * 100:.2f}%')
    print(f'Точность (Precision) гибридной модели на тестовой выборке: {precision * 100:.2f}%')
    print(f'Полнота гибридной модели на тестовой выборке: {recall * 100:.2f}%')
    print(f'F1-мера на тестовой выборке: {f1 * 100:.2f}%')

    # Построение ROC-кривой
    # plot_roc_curve(y_test_list, y_scores)

    return bayesian_output

# Функция для извлечения признаков из URL и IP
def extract_url_features(url, ip):
    # Получаем длину URL
    length = len(url)
    # Подсчитываем количество специальных символов в URL
    num_special_chars = len(re.findall(r'[/?=&\-]', url))
    # Проверяем, начинается ли URL с 'https'
    has_https = 1 if url.startswith('https') else 0
    # Добавляем признак IP, где 1 означает фишинговый IP
    return length, num_special_chars, has_https, ip

# 1. Реализация RNN с использованием PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# 2. Байесовский вариационный метод
class BayesianInfer(nn.Module):
    def __init__(self, input_size, output_size):
        super(BayesianInfer, self).__init__()
        self.correlation_coeff = nn.Parameter(torch.zeros(input_size, input_size))
        self.output_size = output_size

    def forward(self, x):
        # Вычисляем коэффициент корреляции Пирсона
        correlation_matrix = torch.tanh(self.correlation_coeff)
        correlation_coefficients = correlation_matrix.triu(diagonal=1)

        # Аппроксимация байесовской модели с помощью нормального распределения
        mean = torch.zeros_like(x)
        variance = torch.ones_like(x)
        epsilon = torch.randn_like(x)
        z = mean + torch.sqrt(variance) * epsilon
        return torch.sigmoid(z)

# 3. Интеграция RNN и байесовской сети
class HybridModel(nn.Module):
    def __init__(self, rnn_model, bayesian_infer):
        super(HybridModel, self).__init__()
        self.rnn_model = rnn_model
        self.bayesian_infer = bayesian_infer

    def forward(self, x):
        rnn_output = self.rnn_model(x)
        bayesian_output = self.bayesian_infer(rnn_output)
        return bayesian_output

# 4. Загрузка данных из CSV и предобработка
def load_and_preprocess_data(csv_file, test_size=0.2):
    df = pd.read_csv(csv_file)

    # Извлекаем признаки из столбца 'url' и 'ip'
    df['length_url'], df['num_special_chars'], df['has_https'], df['ip'] = zip(*df.apply(lambda row: extract_url_features(row['url'], row['ip']), axis=1))

    # Выделяем признаки и метки классов
    X = df[['length_url', 'num_special_chars', 'has_https', 'ip']].values
    y = df['status'].apply(lambda x: 1 if x == 'phishing' else 0).values  # Преобразуем метки в 0 и 1

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # def balance_data(X_train, y_train):
    #     smote = SMOTE(random_state=54)
    #     X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    #     return X_train_balanced, y_train_balanced

    # Нормализуем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Преобразуем данные в формат PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Добавляем измерение для временных шагов
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)  # Добавляем измерение для временных шагов
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# 5. Обучение и тестирование модели
def train_rnn_model(rnn_model, train_loader, criterion, optimizer, num_epochs=10):
    # Переводим модель в режим обучения
    rnn_model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            # Переводим данные на устройство (GPU или CPU)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Обнуляем градиенты оптимизатора
            optimizer.zero_grad()
            # Получаем выходные данные модели и вычисляем функцию потерь
            outputs = rnn_model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            # Выполняем обратное распространение ошибки и обновляем параметры
            loss.backward()
            optimizer.step()
        print(f'Эпоха [{epoch + 1}/{num_epochs}], Потеря: {loss.item():.4f}')

def test_rnn_model(rnn_model, test_loader):
    # Переводим модель в режим оценки
    rnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = rnn_model(X_batch)
            bayesian_result = torch.sigmoid(outputs)  # Рассчитать результаты байесовского метода
            predicted = (bayesian_result > 0.5).float().squeeze()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    # Вычисляем точность модели
    accuracy = correct / total
    print(correct, total)
    print(f'Точность на тестовой выборке: {accuracy * 100:.2f}%')


# 6. Главная функция для обучения и тестирования
if __name__ == "__main__":
    # Задаем параметры модели
    input_size = 4  # Количество признаков: Длина URL, Спец.символы, HTTPS, IP
    hidden_size = 128  # Увеличиваем размер скрытого слоя
    output_size = 1
    batch_size = 32
    num_epochs = 10  # Увеличиваем количество эпох
    learning_rate = 0.001


    # Определяем устройство для выполнения вычислений (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем и предобрабатываем данные
    train_X, train_y, test_X, test_y = load_and_preprocess_data('dataset_phishing.csv', test_size=0.2)

    # Создаем наборы данных для обучения и тестирования
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Создаем модель RNN
    rnn_model = RNNModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

    # Определяем функцию потерь и оптимизатор
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

    # Обучаем модель
    train_rnn_model(rnn_model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Тестируем модель
    test_accuracy = test_rnn_model(rnn_model, test_loader)

    # Создаем и используем байесовскую сеть
    bayesian_model = BayesianInfer(input_size=4, output_size=4)
    hybrid_model = HybridModel(bayesian_model, rnn_model)

    # Тестируем гибридную модель
    test_hybrid_model(hybrid_model, test_loader)

    # Проверка URL вручную
    while True:
        user_url = input("Введите URL для проверки (или 'exit' для выхода): ")
        if user_url.lower() == 'exit':
            break
        user_ip = input("Введите IP-адрес (или оставьте пустым): ")
        user_ip = int(user_ip) if user_ip else 0  # Преобразуем IP в 0, если не введен
        check_url(hybrid_model, user_url, user_ip)
