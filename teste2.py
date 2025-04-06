import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


# gerar CDF: falhas x vibração, pressão, voltagem, rotação
# Carregar os arquivos para análise
errors_df = pd.read_csv('./PdM_errors.csv')
failures_df = pd.read_csv('./PdM_failures.csv')
machines_df = pd.read_csv('./PdM_machines.csv')
maint_df = pd.read_csv('./PdM_maint.csv')
telemetry_df = pd.read_csv('./PdM_telemetry.csv')

# Garantir que as colunas 'datetime' estejam no formato de data
telemetry_df['datetime'] = pd.to_datetime(telemetry_df['datetime'])
failures_df['datetime'] = pd.to_datetime(failures_df['datetime'])

# Fazer o merge com base em 'machineID' e 'datetime' para associar os dados de telemetria às falhas
merged_df = pd.merge(telemetry_df, failures_df[['datetime', 'machineID', 'failure']], on=['machineID', 'datetime'], how='left')

# Criar uma coluna 'failure_type' que contém o tipo de falha, se houver, e 'failure_count' para falhas (1 = falha, 0 = sem falha)
merged_df['failure_type'] = merged_df['failure'].fillna('No Failure')
merged_df['failure_count'] = merged_df['failure'].notna().astype(int)

# Função para calcular e plotar a CDF usando ECDF do statsmodels
def plot_cdf_ecdf(data, label):
    ecdf = ECDF(data)
    plt.plot(ecdf.x, ecdf.y, label=label)

# Gráfico CDF de Pressão por Tipo de Falha
plt.figure(figsize=(10, 6))
for failure_type in merged_df['failure_type'].unique():
    data = merged_df[merged_df['failure_type'] == failure_type]['pressure']
    plot_cdf_ecdf(data, f'Falha: {failure_type}')

plt.title('CDF de Pressão por Tipo de Falha')
plt.xlabel('Pressão')
plt.ylabel('Probabilidade Acumulada')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico CDF de Vibração por Tipo de Falha
plt.figure(figsize=(10, 6))
for failure_type in merged_df['failure_type'].unique():
    data = merged_df[merged_df['failure_type'] == failure_type]['vibration']
    plot_cdf_ecdf(data, f'Falha: {failure_type}')

plt.title('CDF de Vibração por Tipo de Falha')
plt.xlabel('Vibração')
plt.ylabel('Probabilidade Acumulada')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico CDF de Rotação por Tipo de Falha
plt.figure(figsize=(10, 6))
for failure_type in merged_df['failure_type'].unique():
    data = merged_df[merged_df['failure_type'] == failure_type]['rotate']
    plot_cdf_ecdf(data, f'Falha: {failure_type}')

plt.title('CDF de Rotação por Tipo de Falha')
plt.xlabel('Rotação')
plt.ylabel('Probabilidade Acumulada')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico CDF de Voltagem por Tipo de Falha
plt.figure(figsize=(10, 6))
for failure_type in merged_df['failure_type'].unique():
    data = merged_df[merged_df['failure_type'] == failure_type]['volt']
    plot_cdf_ecdf(data, f'Falha: {failure_type}')

plt.title('CDF de Voltagem por Tipo de Falha')
plt.xlabel('Voltagem')
plt.ylabel('Probabilidade Acumulada')
plt.legend()
plt.grid(True)
plt.show()


### verificar qual hora tem maior falhas
merged_df['hour'] = merged_df['datetime'].dt.hour
failures_by_hour = merged_df.groupby('hour')['failure_count'].sum()

# Plotar falhas por hora do dia
plt.figure(figsize=(10, 6))
plt.plot(failures_by_hour.index, failures_by_hour.values, marker='o')
plt.title('Falhas por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Falhas')
plt.grid(True)
plt.show()

# Converter a coluna 'datetime' para o formato datetime
errors_df['datetime'] = pd.to_datetime(errors_df['datetime'])

# Extrair a hora do dia
errors_df['hour'] = errors_df['datetime'].dt.hour

# Contar o número de erros por hora
errors_by_hour = errors_df.groupby('hour')['errorID'].count()

# Plotar o gráfico de erros por hora do dia
plt.figure(figsize=(10, 6))
plt.plot(errors_by_hour.index, errors_by_hour.values, marker='o')
plt.title('Erros por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Erros')
plt.grid(True)
plt.show()

# Converter a coluna 'datetime' de 'maint_df' para o formato datetime
maint_df['datetime'] = pd.to_datetime(maint_df['datetime'])

# Extrair a hora do dia
maint_df['hour'] = maint_df['datetime'].dt.hour

# Contar o número de manutenções por hora
maint_by_hour = maint_df.groupby('hour')['machineID'].count()




####

# Contar o número total de falhas por machineID
total_failures_by_machine = failures_df.groupby('machineID')['failure'].count()

# Plotar o gráfico de barras
plt.figure(figsize=(10, 6))
total_failures_by_machine.plot(kind='bar', color='blue')
plt.title('Número Total de Falhas por Máquina')
plt.xlabel('ID da Máquina')
plt.ylabel('Número de Falhas')
plt.grid(True)
plt.show()


#####
# verificar qual é a predominância da falha no machineID

# Agrupar o número de falhas por machineID e por tipo de falha
failures_by_machine_type = failures_df.groupby(['machineID', 'failure']).size().unstack(fill_value=0)

# Identificar o tipo de falha mais frequente (predominante) para cada máquina
predominant_failure_by_machine = failures_by_machine_type.idxmax(axis=1)

# Criar uma lista com os IDs das máquinas e os tipos de falhas predominantes para o gráfico
machines = list(predominant_failure_by_machine.index)
failures = list(predominant_failure_by_machine.values)

# Converter o tipo de falha em valores numéricos para o gráfico
failure_mapping = {'comp1': 1, 'comp2': 2, 'comp3': 3, 'comp4': 4}
failure_numeric = [failure_mapping[f] for f in failures]

# Plotar o gráfico de barras com os IDs das máquinas e os tipos de falhas predominantes
plt.figure(figsize=(14, 6))
plt.bar(machines, failure_numeric, color='blue', edgecolor='black')

# Adicionar rótulos no eixo Y
plt.yticks([1, 2, 3, 4], ['comp1', 'comp2', 'comp3', 'comp4'])

# Adicionar todos os rótulos no eixo X com rotação para melhorar a legibilidade
plt.xticks(machines, rotation=90, fontsize=10)

# Definir título e rótulos dos eixos
plt.title('Tipo de Falha Predominante por Máquina (1 a 100)', fontsize=16)
plt.xlabel('ID da Máquina', fontsize=12)
plt.ylabel('Tipo de Falha Predominante', fontsize=12)

# Adicionar grid
plt.grid(True)

# Ajustar layout para não cortar os rótulos
plt.tight_layout()

# Exibir o gráfico
plt.show()