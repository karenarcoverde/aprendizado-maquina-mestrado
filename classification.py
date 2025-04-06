import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Carregar os arquivos
machines = pd.read_csv('./PdM_machines.csv')
maint = pd.read_csv('./PdM_maint.csv')
telemetry = pd.read_csv('./PdM_telemetry.csv')
errors = pd.read_csv('./PdM_errors.csv')
failures = pd.read_csv('./PdM_failures.csv')

# Converter as colunas datetime para formato de data
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
errors['datetime'] = pd.to_datetime(errors['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])

# Agregar os dados de telemetria por hora para reduzir duplicação
telemetry = telemetry.groupby(['machineID', pd.Grouper(key='datetime', freq='1H')]).mean().reset_index()

# Merge com dados de máquinas
data_combined = telemetry.merge(machines, on='machineID', how='left')

# Adicionar informações de falhas, mantendo apenas falhas próximas ao tempo da telemetria
failures['failure_time'] = failures['datetime']
data_combined = data_combined.merge(failures[['machineID', 'failure_time']], 
                                    on='machineID', how='left')
data_combined['failure_label'] = (data_combined['datetime'] >= data_combined['failure_time']).astype(int)

# Limitar erros e manutenção para períodos relevantes
errors = errors[errors['datetime'].between(data_combined['datetime'].min(), data_combined['datetime'].max())]
maint = maint[maint['datetime'].between(data_combined['datetime'].min(), data_combined['datetime'].max())]

# Criar features
features = ['volt', 'rotate', 'pressure', 'vibration', 'age']
data_combined = data_combined.dropna(subset=features)
X = data_combined[features]
y = data_combined['failure_label']

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Treinar o modelo
# model = RandomForestClassifier(
#     n_estimators=50,       # Número menor de árvores
#     max_depth=10,          # Limite de profundidade das árvores
#     random_state=42,
#     n_jobs=-1              # Utiliza todos os núcleos
# )

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    min_child_weight=5,
    random_state=42
)

model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salvar o modelo
import joblib
joblib.dump(model, 'failure_prediction_model.pkl')

# Adicionar análise detalhada dos resultados
results = X_test.copy()  # Copiar as entradas do conjunto de teste
results['Real'] = y_test.values  # Adicionar valores reais
results['Predito'] = y_pred  # Adicionar valores previstos

# Mostrar amostra dos resultados
print("Amostra dos Resultados:")
print(results.head(10))

# Salvar resultados em CSV com as entradas (features)
results.to_csv('predictions_with_inputs.csv', index=False)
print("Resultados salvos em 'predictions_with_inputs.csv'")
