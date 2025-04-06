import pandas as pd

# Carregar o arquivo CSV
file_path = './PdM_telemetry.csv'
data = pd.read_csv(file_path)

# Converter a coluna 'datetime' para datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Criar uma nova coluna com a média dos parâmetros (volt, rotate, pressure, vibration) por máquina
agg_data = data.groupby('machineID').agg({
    'volt': 'mean',
    'rotate': 'mean',
    'pressure': 'mean',
    'vibration': 'mean'
}).reset_index()

# Renomear as colunas para indicar que são médias
agg_data.columns = ['machineID', 'mean_volt', 'mean_rotate', 'mean_pressure', 'mean_vibration']

# Salvar o resultado em um novo arquivo CSV
output_path = './PdM_telemetry_aggregated.csv'
agg_data.to_csv(output_path, index=False)

print(f"Arquivo salvo em {output_path}")
