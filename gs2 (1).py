import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DonationClimateSystem:
    """
    Sistema integrado de ML para gerenciamento inteligente de doações
    baseado em dados climáticos
    """
    
    def __init__(self):
        self.geographic_clusters = None
        self.scaler_geo = StandardScaler()
        self.scaler_temp = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.need_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cluster_profiles = {}
        
    def load_and_prepare_data(self, df):
        """Carrega e prepara os dados para análise"""
        print("🔄 Preparando dados climáticos...")
        
        # Converter data
        df['measure_date'] = pd.to_datetime(df['measure_date'])
        df['month'] = df['measure_date'].dt.month
        df['season'] = df['month'].apply(self._get_season)
        
        # Calcular métricas climáticas
        df['temp_amplitude'] = df['max_temperature'] - df['min_temperature']
        df['temp_average'] = (df['max_temperature'] + df['min_temperature']) / 2
        
        # Imputar valores faltantes
        temp_cols = ['max_temperature', 'min_temperature', 'temp_amplitude', 'temp_average']
        df[temp_cols] = self.imputer.fit_transform(df[temp_cols])
        
        print(f"✅ Dados preparados: {len(df)} registros processados")
        return df
    
    def _get_season(self, month):
        """Determina a estação do ano (hemisfério sul)"""
        if month in [12, 1, 2]:
            return 'Verão'
        elif month in [3, 4, 5]:
            return 'Outono'
        elif month in [6, 7, 8]:
            return 'Inverno'
        else:
            return 'Primavera'
    
    def geographic_segmentation(self, df):
        """Segmentação geográfica baseada em padrões climáticos"""
        print("🗺️  Executando segmentação geográfica...")
        
        # Agregar dados por localização
        location_features = df.groupby(['station_name', 'latitude', 'longitude', 'altitude']).agg({
            'max_temperature': ['mean', 'std', 'min', 'max'],
            'min_temperature': ['mean', 'std', 'min', 'max'],
            'temp_amplitude': ['mean', 'std'],
            'temp_average': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        location_features.columns = ['_'.join(col).strip() for col in location_features.columns]
        location_features = location_features.reset_index()
        
        # Preparar features para clustering
        cluster_features = location_features[[
            'latitude', 'longitude', 'altitude',
            'max_temperature_mean', 'min_temperature_mean',
            'temp_amplitude_mean', 'temp_average_mean',
            'max_temperature_std', 'min_temperature_std'
        ]].fillna(0)
        
        # Normalizar features
        features_scaled = self.scaler_geo.fit_transform(cluster_features)
        
        # Encontrar número ótimo de clusters
        best_k = self._find_optimal_clusters(features_scaled)
        
        # Aplicar K-means
        self.geographic_clusters = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = self.geographic_clusters.fit_predict(features_scaled)
        
        location_features['cluster'] = clusters
        
        # Criar perfis dos clusters
        self._create_cluster_profiles(location_features)
        
        print(f"✅ Segmentação concluída: {best_k} clusters identificados")
        return location_features
    
    def _find_optimal_clusters(self, features, max_k=8):
        """Encontra o número ótimo de clusters usando silhouette score"""
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"📊 Número ótimo de clusters: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")
        return optimal_k
    
    def _create_cluster_profiles(self, location_features):
        """Cria perfis detalhados de cada cluster"""
        for cluster_id in location_features['cluster'].unique():
            cluster_data = location_features[location_features['cluster'] == cluster_id]
            
            profile = {
                'nome': f'Região Climática {cluster_id + 1}',
                'localizations': len(cluster_data),
                'temp_media': cluster_data['temp_average_mean'].mean(),
                'temp_max_media': cluster_data['max_temperature_mean'].mean(),
                'temp_min_media': cluster_data['min_temperature_mean'].mean(),
                'amplitude_media': cluster_data['temp_amplitude_mean'].mean(),
                'altitude_media': cluster_data['altitude'].mean(),
                'coordenadas_centro': {
                    'lat': cluster_data['latitude'].mean(),
                    'lng': cluster_data['longitude'].mean()
                }
            }
            
            # Determinar características climáticas
            if profile['temp_media'] > 25:
                profile['caracteristica'] = 'Região Quente'
                profile['necessidades_priorizadas'] = ['Ventiladores', 'Água', 'Roupas leves', 'Protetor solar']
            elif profile['temp_media'] < 18:
                profile['caracteristica'] = 'Região Fria'
                profile['necessidades_priorizadas'] = ['Agasalhos', 'Cobertores', 'Aquecedores', 'Alimentos quentes']
            else:
                profile['caracteristica'] = 'Região Temperada'
                profile['necessidades_priorizadas'] = ['Roupas variadas', 'Medicamentos', 'Alimentos gerais']
            
            self.cluster_profiles[cluster_id] = profile
    
    def create_need_thermometer(self, df):
        """Cria o modelo do Termômetro de Necessidade"""
        print("🌡️  Construindo Termômetro de Necessidade...")
        
        # Criar índice de necessidade baseado em condições extremas
        df['heat_stress'] = np.where(df['max_temperature'] > 35, 
                                   (df['max_temperature'] - 35) * 2, 0)
        df['cold_stress'] = np.where(df['min_temperature'] < 10, 
                                   (10 - df['min_temperature'] * 3), 0)
        df['temp_variation_stress'] = np.where(df['temp_amplitude'] > 15,
                                             (df['temp_amplitude'] - 15), 0)
        
        # Índice de necessidade (0-100)
        df['need_index'] = np.clip(
            df['heat_stress'] + df['cold_stress'] + df['temp_variation_stress'],
            0, 100
        )
        
        # Preparar features para o modelo
        features = ['max_temperature', 'min_temperature', 'temp_amplitude', 
                   'temp_average', 'altitude', 'month']
        
        X = df[features].fillna(df[features].mean())
        y = df['need_index']
        
        # Treinar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.need_predictor.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.need_predictor.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"✅ Termômetro calibrado - R²: {r2:.3f}, RMSE: {rmse:.2f}")
        
        return df
    
    def predict_regional_needs(self, location_data, weather_forecast):
        """Prediz necessidades regionais baseado na previsão do tempo"""
        predictions = []
        
        for _, location in location_data.iterrows():
            cluster_id = location['cluster']
            cluster_profile = self.cluster_profiles[cluster_id]
            
            # Predizer índice de necessidade
            features = np.array([[
                weather_forecast['max_temp'],
                weather_forecast['min_temp'],
                weather_forecast['max_temp'] - weather_forecast['min_temp'],
                (weather_forecast['max_temp'] + weather_forecast['min_temp']) / 2,
                location['altitude'],
                weather_forecast['month']
            ]])
            
            need_score = self.need_predictor.predict(features)[0]
            
            # Classificar nível de urgência
            if need_score > 70:
                urgency = 'CRÍTICA'
                color = '#FF4444'
            elif need_score > 40:
                urgency = 'ALTA'
                color = '#FF8800'
            elif need_score > 20:
                urgency = 'MODERADA'
                color = '#FFBB00'
            else:
                urgency = 'BAIXA'
                color = '#44AA44'
            
            predictions.append({
                'regiao': location['station_name'],
                'cluster': cluster_profile['nome'],
                'necessidade_score': round(need_score, 1),
                'urgencia': urgency,
                'cor_alerta': color,
                'itens_prioritarios': cluster_profile['necessidades_priorizadas'],
                'coordenadas': {
                    'lat': location['latitude'],
                    'lng': location['longitude']
                }
            })
        
        return sorted(predictions, key=lambda x: x['necessidade_score'], reverse=True)
    
    def generate_insights_report(self, location_data):
        """Gera relatório de insights para estratégias de doação"""
        report = {
            'resumo_clusters': {},
            'recomendacoes_estrategicas': [],
            'alertas_sazonais': []
        }
        
        # Resumo dos clusters
        for cluster_id, profile in self.cluster_profiles.items():
            report['resumo_clusters'][f'cluster_{cluster_id}'] = {
                'nome': profile['nome'],
                'caracteristica': profile['caracteristica'],
                'localizacoes': profile['localizations'],
                'temperatura_media': round(profile['temp_media'], 1),
                'necessidades': profile['necessidades_priorizadas']
            }
        
        # Recomendações estratégicas
        hot_clusters = [p for p in self.cluster_profiles.values() if p['temp_media'] > 25]
        cold_clusters = [p for p in self.cluster_profiles.values() if p['temp_media'] < 18]
        
        if hot_clusters:
            report['recomendacoes_estrategicas'].append(
                f"🔥 {len(hot_clusters)} regiões quentes identificadas - priorizar ventiladores e água"
            )
        
        if cold_clusters:
            report['recomendacoes_estrategicas'].append(
                f"❄️ {len(cold_clusters)} regiões frias identificadas - priorizar agasalhos e cobertores"
            )
        
        # Alertas sazonais
        report['alertas_sazonais'] = [
            "🌡️ Verão (Dez-Fev): Aumentar estoque de ventiladores e água nas regiões quentes",
            "🧥 Inverno (Jun-Ago): Intensificar campanhas de agasalhos nas regiões frias",
            "⚠️ Transições (Mar-Mai, Set-Nov): Preparar para variações bruscas de temperatura"
        ]
        
        return report

# Exemplo de uso do sistema
def main():
    """Demonstração do sistema completo"""
    
    # Simular dados para demonstração
    np.random.seed(42)
    sample_data = []
    
    stations = [
        # Rio de Janeiro
        ("ALTO DA BOA VISTA", 83007, -22.965833, -43.279167, 347.1),
        ("COPACABANA", 83008, -22.970722, -43.182365, 5.2),
        ("TIJUCA", 83009, -22.925278, -43.238889, 80.0),
        ("BARRA DA TIJUCA", 83010, -23.018056, -43.365833, 2.1),
        
        # São Paulo
        ("MIRANTE DE SANTANA", 83004, -23.503056, -46.618333, 792.1),
        ("CONGONHAS", 83005, -23.627778, -46.656111, 803.7),
        ("IBIRAPUERA", 83006, -23.587500, -46.660833, 733.4),
        
        # Minas Gerais
        ("BELO HORIZONTE", 83587, -19.932222, -43.937778, 915.0),
        ("CONFINS", 83595, -19.635556, -43.968056, 827.4),
        
        # Brasília
        ("BRASILIA DF", 83377, -15.789444, -47.925833, 1159.5),
        
        # Bahia
        ("SALVADOR", 83229, -12.910833, -38.331667, 51.4),
        
        # Ceará
        ("FORTALEZA", 82397, -3.776389, -38.532222, 26.5),
        
        # Pernambuco
        ("RECIFE", 82900, -8.050556, -34.950833, 10.4),
        
        # Amazonas
        ("MANAUS", 82331, -3.102778, -60.016667, 67.0),
        
        # Pará
        ("BELEM", 82191, -1.379167, -48.476111, 10.8),
        
        # Rio Grande do Sul
        ("PORTO ALEGRE", 83967, -30.053056, -51.178611, 46.97),
        ("CAXIAS DO SUL", 83914, -29.197222, -51.187500, 760.0),
        
        # Santa Catarina
        ("FLORIANOPOLIS", 83897, -27.583056, -48.565833, 1.84),
        
        # Paraná
        ("CURITIBA", 83842, -25.448056, -49.231944, 923.5),
        
        # Goiás
        ("GOIANIA", 83423, -16.632222, -49.220556, 741.5),
        
        # Mato Grosso
        ("CUIABA", 83361, -15.652778, -56.116667, 165.2),
        
        # Espírito Santo
        ("VITORIA", 83648, -20.315556, -40.319167, 36.2),
        
        # Sergipe
        ("ARACAJU", 83096, -10.950000, -37.043333, 4.72),
        
        # Alagoas
        ("MACEIO", 82994, -9.666944, -35.735278, 64.5),
        
        # Paraíba
        ("JOAO PESSOA", 82798, -7.110556, -34.863056, 7.4),
        
        # Rio Grande do Norte
        ("NATAL", 82599, -5.911111, -35.248056, 48.6),
        
        # Piauí
        ("TERESINA", 82578, -5.089167, -42.823611, 72.7),
        
        # Maranhão
        ("SAO LUIS", 82280, -2.602500, -44.214167, 51.0),
        
        # Tocantins
        ("PALMAS", 83235, -10.290278, -48.357500, 280.0),
        
        # Acre
        ("RIO BRANCO", 82915, -9.958889, -67.800833, 160.0),
        
        # Rondônia
        ("PORTO VELHO", 82825, -8.761944, -63.902500, 95.0),
        
        # Roraima
        ("BOA VISTA", 82022, 2.820556, -60.671389, 90.0),
        
        # Amapá
        ("MACAPA", 82098, 0.050556, -51.066111, 14.5),
        
        # Mato Grosso do Sul
        ("CAMPO GRANDE", 83612, -20.469444, -54.613889, 532.1)
    ]
    
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    for date in dates[:2000]:  # Aumentando para mais dados de demonstração
        for station_name, station_id, lat, lng, alt in stations:
            # Simular temperaturas baseadas na região e estação
            month = date.month
            lat = lat  # latitude da estação
            
            # Ajustar temperatura base pela latitude (mais ao norte = mais quente)
            if lat > -10:  # Norte/Nordeste
                base_summer, base_winter = 35, 28
                var_summer, var_winter = 3, 2
            elif lat > -20:  # Centro-Oeste/Nordeste
                base_summer, base_winter = 32, 25
                var_summer, var_winter = 4, 3
            elif lat > -25:  # Sudeste
                base_summer, base_winter = 30, 22
                var_summer, var_winter = 4, 4
            else:  # Sul
                base_summer, base_winter = 28, 18
                var_summer, var_winter = 3, 5
            
            # Ajustar pela altitude (cada 100m = -0.6°C)
            altitude_adj = -alt * 0.006
            
            if month in [12, 1, 2]:  # Verão
                max_temp = np.random.normal(base_summer + altitude_adj, var_summer)
                min_temp = np.random.normal(base_summer - 8 + altitude_adj, var_summer-1)
            elif month in [6, 7, 8]:  # Inverno
                max_temp = np.random.normal(base_winter + altitude_adj, var_winter)
                min_temp = np.random.normal(base_winter - 7 + altitude_adj, var_winter-1)
            else:  # Outono/Primavera
                avg_temp = (base_summer + base_winter) / 2
                max_temp = np.random.normal(avg_temp + altitude_adj, (var_summer + var_winter)/2)
                min_temp = np.random.normal(avg_temp - 8 + altitude_adj, (var_summer + var_winter)/2)
            
            # Adicionar alguns valores faltantes
            if np.random.random() < 0.05:
                max_temp = np.nan
            if np.random.random() < 0.05:
                min_temp = np.nan
                
            sample_data.append({
                'station_name': station_name,
                'station_id': station_id,
                'latitude': lat,
                'longitude': lng,
                'altitude': alt,
                'measure_date': date,
                'max_temperature': max_temp,
                'min_temperature': min_temp
            })
    
    df = pd.DataFrame(sample_data)
    
    print("🚀 SISTEMA DE DOAÇÕES INTELIGENTE")
    print("=" * 50)
    
    # Inicializar sistema
    system = DonationClimateSystem()
    
    # Preparar dados
    df_prepared = system.load_and_prepare_data(df)
    
    # Executar segmentação geográfica
    location_clusters = system.geographic_segmentation(df_prepared)
    
    # Construir termômetro de necessidade
    df_with_needs = system.create_need_thermometer(df_prepared)
    
    # Gerar previsões para uma situação exemplo
    weather_forecast = {
        'max_temp': 38,  # Onda de calor
        'min_temp': 28,
        'month': 1  # Janeiro
    }
    
    print("\n🌡️ TERMÔMETRO DE NECESSIDADE - ONDA DE CALOR")
    print("=" * 50)
    
    predictions = system.predict_regional_needs(location_clusters, weather_forecast)
    
    for pred in predictions:
        print(f"\n📍 {pred['regiao']}")
        print(f"   Cluster: {pred['cluster']}")
        print(f"   Necessidade: {pred['necessidade_score']}/100 ({pred['urgencia']})")
        print(f"   Itens prioritários: {', '.join(pred['itens_prioritarios'])}")
    
    # Gerar relatório de insights
    print("\n📊 RELATÓRIO DE INSIGHTS ESTRATÉGICOS")
    print("=" * 50)
    
    insights = system.generate_insights_report(location_clusters)
    
    print("\n🗺️ Resumo dos Clusters:")
    for cluster_key, cluster_info in insights['resumo_clusters'].items():
        print(f"  • {cluster_info['nome']}: {cluster_info['caracteristica']} "
              f"({cluster_info['localizacoes']} locais, {cluster_info['temperatura_media']}°C)")
    
    print("\n💡 Recomendações Estratégicas:")
    for rec in insights['recomendacoes_estrategicas']:
        print(f"  • {rec}")
    
    print("\n⏰ Alertas Sazonais:")
    for alert in insights['alertas_sazonais']:
        print(f"  • {alert}")
    
    print("\n✅ Sistema implementado com sucesso!")
    print("📱 Pronto para integração no app de doações!")

if __name__ == "__main__":
    main()
