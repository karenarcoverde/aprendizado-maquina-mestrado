import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Carregar os dados
telemetry = pd.read_csv('./PdM_telemetry.csv')
failures = pd.read_csv('./PdM_failures.csv')

# Converter colunas de data para o tipo datetime
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
failures['failure_date'] = pd.to_datetime(failures['datetime'])

# Mesclar os dados de falhas com o dataset de telemetria
merged_data = telemetry.merge(failures[['machineID', 'failure_date']], on="machineID", how="left")

# Substituir valores nulos na coluna de falha com uma data futura
merged_data['failure_date'] = merged_data['failure_date'].fillna(pd.Timestamp.now() + pd.DateOffset(days=30))

# Calcular o tempo até a falha em horas
merged_data['time_to_failure'] = (merged_data['failure_date'] - merged_data['datetime']).dt.total_seconds() / 3600

# Verificar se há valores negativos antes de aplicar log
merged_data = merged_data[merged_data['time_to_failure'] > 0]

# Aplicar transformação logarítmica para normalizar a saída
merged_data['time_to_failure'] = np.log1p(merged_data['time_to_failure'])

# Amostrar uma porcentagem dos dados para treinamento mais rápido
merged_data = merged_data.sample(frac=0.1, random_state=42)

# Separar as variáveis de entrada e saída
X = merged_data.drop(columns=['time_to_failure', 'datetime', 'failure_date'])
y = merged_data['time_to_failure']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar e remover linhas com NaN ou valores infinitos
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.replace([np.inf, -np.inf], np.nan).dropna()
y_test = y_test.replace([np.inf, -np.inf], np.nan).dropna()

# Escalonar os dados
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

# Reverter a transformação logarítmica da variável de saída para voltar à escala original
merged_data['time_to_failure_original'] = np.expm1(merged_data['time_to_failure'])

# Listar as variáveis de entrada
input_vars = X.columns

# Criar um gráfico de dispersão para cada variável de entrada em relação à saída original
plt.figure(figsize=(15, 20))
for i, var in enumerate(input_vars, 1):
    plt.subplot(len(input_vars) // 2 + 1, 2, i)
    sns.scatterplot(data=merged_data, x=var, y='time_to_failure_original', alpha=0.5)
    plt.title(f'Relação entre {var} e Tempo até a Falha (Escala Original)')
    plt.xlabel(var)
    plt.ylabel('Tempo até a Falha')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Contar as falhas de cada componente para cada máquina
comp_failures = failures.groupby(['machineID', 'failure']).size().unstack(fill_value=0)
comp_failures.columns = [f'{col}_failures' for col in comp_failures.columns]
comp_failures.reset_index(inplace=True)

# Calcular as médias das métricas de telemetria para cada máquina
telemetry_means = telemetry.groupby('machineID')[['vibration', 'volt', 'pressure', 'rotate']].mean().reset_index()

# Mesclar as contagens de falhas com as médias de telemetria
merged_analysis = pd.merge(comp_failures, telemetry_means, on='machineID', how='left')

# Listar componentes e variáveis de telemetria para criar os gráficos
components = comp_failures.columns[1:]  # Ignorando a coluna 'machineID'
telemetry_vars = telemetry_means.columns[1:]  # Ignorando a coluna 'machineID'

# Criar gráficos de dispersão para cada componente em relação a cada métrica de telemetria
plt.figure(figsize=(20, 15))
plot_num = 1
for comp in components:
    for var in telemetry_vars:
        plt.subplot(len(components), len(telemetry_vars), plot_num)
        sns.scatterplot(data=merged_analysis, x=comp, y=var, alpha=0.5)
        plt.title(f'Relação entre Falhas de {comp} e {var}')
        plt.xlabel(f'Falhas de {comp}')
        plt.ylabel(var)
        plot_num += 1

plt.tight_layout()
plt.show()


# Construir uma rede neural mais complexa com Batch Normalization
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

# Compilar o modelo com MSE e MAPE como métricas adicionais
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mape'])

# Configurar callback para parar o treinamento cedo se o modelo começar a overfit
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinar o modelo com mais épocas e salvar o histórico de treinamento
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Exibir o Loss, MSE e MAPE para cada época
for epoch, (loss, mse, mape) in enumerate(zip(history.history['loss'], history.history['mse'], history.history['mape']), start=1):
    print(f'Época {epoch}: Loss = {loss:.4f}, MSE = {mse:.4f}, MAPE = {mape:.4f}')

# Fazer previsões
predictions = model.predict(X_test)

# Reverter a transformação logarítmica para comparação com os valores reais
y_test_exp = np.expm1(y_test)
predictions_exp = np.expm1(predictions)

# Calcular MSE no conjunto de teste
mse_exp = mean_squared_error(y_test_exp, predictions_exp)

print(f'\nMSE no conjunto de teste (escala original): {mse_exp}')

# Exibir as previsões junto com os valores reais
results = pd.DataFrame({'Real': y_test_exp, 'Previsto': predictions_exp.flatten()})
print(results.head())


