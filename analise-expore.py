import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SolidárIA_EDA:
    """
    🤖 SolidárIA - Análise Exploratória de Dados
    
    Módulo de análise inteligente para descobrir padrões climáticos
    que otimizam estratégias de doação e identificam necessidades regionais.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
        
    def prepare_data(self):
        """Prepara os dados para análise"""
        print("🤖 SolidárIA preparando dados para análise inteligente...")
        
        # Converter data e extrair features temporais
        self.df['measure_date'] = pd.to_datetime(self.df['measure_date'])
        self.df['year'] = self.df['measure_date'].dt.year
        self.df['month'] = self.df['measure_date'].dt.month
        self.df['day_of_year'] = self.df['measure_date'].dt.dayofyear
        self.df['season'] = self.df['month'].apply(self._get_season)
        
        # Calcular métricas climáticas
        self.df['temp_amplitude'] = self.df['max_temperature'] - self.df['min_temperature']
        self.df['temp_average'] = (self.df['max_temperature'] + self.df['min_temperature']) / 2
        
        # Categorizar regiões
        self.df['region'] = self.df.apply(self._categorize_region, axis=1)
        
        # Calcular índices de necessidade
        self.df['heat_stress'] = np.where(self.df['max_temperature'] > 35, 
                                        (self.df['max_temperature'] - 35) * 2, 0)
        self.df['cold_stress'] = np.where(self.df['min_temperature'] < 10, 
                                        (10 - self.df['min_temperature']) * 3, 0)
        self.df['need_index'] = np.clip(self.df['heat_stress'] + self.df['cold_stress'], 0, 100)
        
        print(f"🤖 SolidárIA processou: {len(self.df)} registros, {self.df['station_name'].nunique()} estações")
    
    def _get_season(self, month):
        """Determina a estação do ano (hemisfério sul)"""
        seasons = {
            12: 'Verão', 1: 'Verão', 2: 'Verão',
            3: 'Outono', 4: 'Outono', 5: 'Outono',
            6: 'Inverno', 7: 'Inverno', 8: 'Inverno',
            9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
        }
        return seasons[month]
    
    def _categorize_region(self, row):
        """Categoriza a região baseada na latitude"""
        lat = row['latitude']
        if lat > -10:
            return 'Norte/Nordeste'
        elif lat > -20:
            return 'Centro-Oeste'
        elif lat > -25:
            return 'Sudeste'
        else:
            return 'Sul'
    
    def overview_statistics(self):
        """Estatísticas gerais dos dados"""
        print("\n📊 VISÃO GERAL DOS DADOS")
        print("=" * 50)
        
        print(f"📅 Período: {self.df['measure_date'].min().date()} até {self.df['measure_date'].max().date()}")
        print(f"🗺️  Estações: {self.df['station_name'].nunique()}")
        print(f"📍 Regiões: {self.df['region'].nunique()}")
        print(f"📊 Total de registros: {len(self.df):,}")
        
        # Estatísticas de temperatura
        temp_stats = self.df[['max_temperature', 'min_temperature', 'temp_average', 'temp_amplitude']].describe()
        print(f"\n🌡️  ESTATÍSTICAS DE TEMPERATURA")
        print(temp_stats.round(2))
        
        # Valores faltantes
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  VALORES FALTANTES:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count} ({count/len(self.df)*100:.1f}%)")
        else:
            print(f"\n✅ Nenhum valor faltante encontrado")
    
    def regional_analysis(self):
        """Análise por região"""
        print(f"\n🗺️  ANÁLISE REGIONAL")
        print("=" * 50)
        
        regional_stats = self.df.groupby('region').agg({
            'max_temperature': ['mean', 'std', 'min', 'max'],
            'min_temperature': ['mean', 'std', 'min', 'max'],
            'temp_average': ['mean', 'std'],
            'temp_amplitude': ['mean', 'std'],
            'need_index': ['mean', 'max'],
            'station_name': 'nunique'
        }).round(2)
        
        # Flatten column names
        regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns]
        
        print("📊 Resumo por Região:")
        for region in regional_stats.index:
            stats = regional_stats.loc[region]
            print(f"\n🔸 {region}:")
            print(f"   Estações: {int(stats['station_name_nunique'])}")
            print(f"   Temp. média: {stats['temp_average_mean']:.1f}°C (±{stats['temp_average_std']:.1f})")
            print(f"   Amplitude média: {stats['temp_amplitude_mean']:.1f}°C")
            print(f"   Índice de necessidade: {stats['need_index_mean']:.1f}/100")
        
        return regional_stats
    
    def seasonal_patterns(self):
        """Análise de padrões sazonais"""
        print(f"\n🌀 PADRÕES SAZONAIS")
        print("=" * 50)
        
        seasonal_stats = self.df.groupby(['region', 'season']).agg({
            'temp_average': 'mean',
            'need_index': 'mean'
        }).round(2)
        
        print("🌡️  Temperatura média por estação:")
        temp_pivot = seasonal_stats['temp_average'].unstack()
        print(temp_pivot)
        
        print(f"\n🚨 Índice de necessidade por estação:")
        need_pivot = seasonal_stats['need_index'].unstack()
        print(need_pivot)
        
        return seasonal_stats
    
    def extreme_events_analysis(self):
        """Análise de eventos extremos"""
        print(f"\n🚨 ANÁLISE DE EVENTOS EXTREMOS")
        print("=" * 50)
        
        # Definir limiares
        hot_threshold = 35
        cold_threshold = 10
        high_amplitude_threshold = 20
        
        # Contar eventos extremos
        hot_days = self.df[self.df['max_temperature'] > hot_threshold]
        cold_days = self.df[self.df['min_temperature'] < cold_threshold]
        high_amplitude_days = self.df[self.df['temp_amplitude'] > high_amplitude_threshold]
        
        print(f"🔥 Dias muito quentes (>{hot_threshold}°C): {len(hot_days):,} ({len(hot_days)/len(self.df)*100:.1f}%)")
        print(f"❄️  Dias muito frios (<{cold_threshold}°C): {len(cold_days):,} ({len(cold_days)/len(self.df)*100:.1f}%)")
        print(f"📊 Dias alta amplitude (>{high_amplitude_threshold}°C): {len(high_amplitude_days):,} ({len(high_amplitude_days)/len(self.df)*100:.1f}%)")
        
        # Top estações com eventos extremos
        if len(hot_days) > 0:
            hot_stations = hot_days['station_name'].value_counts().head()
            print(f"\n🔥 Estações com mais dias quentes:")
            for station, count in hot_stations.items():
                print(f"   {station}: {count} dias")
        
        if len(cold_days) > 0:
            cold_stations = cold_days['station_name'].value_counts().head()
            print(f"\n❄️  Estações com mais dias frios:")
            for station, count in cold_stations.items():
                print(f"   {station}: {count} dias")
        
        return {
            'hot_days': hot_days,
            'cold_days': cold_days,
            'high_amplitude_days': high_amplitude_days
        }
    
    def need_index_analysis(self):
        """Análise do índice de necessidade"""
        print(f"\n🎯 ANÁLISE DO ÍNDICE DE NECESSIDADE")
        print("=" * 50)
        
        # Estatísticas do índice
        need_stats = self.df['need_index'].describe()
        print("📊 Estatísticas do Índice de Necessidade:")
        print(need_stats.round(2))
        
        # Classificar por urgência
        self.df['urgency_level'] = pd.cut(
            self.df['need_index'],
            bins=[0, 20, 40, 70, 100],
            labels=['Baixa', 'Moderada', 'Alta', 'Crítica'],
            include_lowest=True
        )
        
        urgency_counts = self.df['urgency_level'].value_counts()
        print(f"\n🚨 Distribuição por Nível de Urgência:")
        for level, count in urgency_counts.items():
            print(f"   {level}: {count:,} dias ({count/len(self.df)*100:.1f}%)")
        
        # Top estações por necessidade média
        station_needs = self.df.groupby('station_name')['need_index'].mean().sort_values(ascending=False)
        print(f"\n🎯 Top 10 Estações por Necessidade Média:")
        for station, need in station_needs.head(10).items():
            print(f"   {station}: {need:.1f}/100")
        
        return station_needs
    
    def create_visualizations(self):
        """Cria visualizações dos dados"""
        print(f"\n📈 GERANDO VISUALIZAÇÕES")
        print("=" * 50)
        
        # Configurar subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('🤖 SolidárIA - Análise Inteligente de Dados Climáticos', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de temperaturas por região
        self.df.boxplot(column='temp_average', by='region', ax=axes[0,0])
        axes[0,0].set_title('Distribuição de Temperatura Média por Região')
        axes[0,0].set_xlabel('Região')
        axes[0,0].set_ylabel('Temperatura Média (°C)')
        
        # 2. Padrões sazonais
        seasonal_temps = self.df.groupby(['season', 'region'])['temp_average'].mean().unstack()
        seasonal_temps.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Temperatura Média por Estação e Região')
        axes[0,1].set_xlabel('Estação')
        axes[0,1].set_ylabel('Temperatura Média (°C)')
        axes[0,1].legend(title='Região', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Distribuição do índice de necessidade
        self.df['need_index'].hist(bins=30, ax=axes[1,0], edgecolor='black', alpha=0.7)
        axes[1,0].set_title('Distribuição do Índice de Necessidade')
        axes[1,0].set_xlabel('Índice de Necessidade')
        axes[1,0].set_ylabel('Frequência')
        
        # 4. Correlação entre variáveis
        corr_vars = ['max_temperature', 'min_temperature', 'temp_amplitude', 'altitude', 'need_index']
        corr_matrix = self.df[corr_vars].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Matriz de Correlação')
        
        # 5. Eventos extremos por mês
        monthly_extremes = self.df.groupby('month').agg({
            'heat_stress': lambda x: (x > 0).sum(),
            'cold_stress': lambda x: (x > 0).sum()
        })
        monthly_extremes.plot(kind='bar', ax=axes[2,0])
        axes[2,0].set_title('Eventos Extremos por Mês')
        axes[2,0].set_xlabel('Mês')
        axes[2,0].set_ylabel('Número de Eventos')
        axes[2,0].legend(['Estresse por Calor', 'Estresse por Frio'])
        
        # 6. Necessidade por região e estação
        need_by_season = self.df.groupby(['region', 'season'])['need_index'].mean().unstack()
        need_by_season.plot(kind='bar', ax=axes[2,1])
        axes[2,1].set_title('Índice de Necessidade por Região e Estação')
        axes[2,1].set_xlabel('Região')
        axes[2,1].set_ylabel('Índice de Necessidade')
        axes[2,1].legend(title='Estação', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Mapa interativo das estações (se plotly estiver disponível)
        try:
            self.create_interactive_map()
        except:
            print("⚠️  Plotly não disponível para mapa interativo")
    
    def create_interactive_map(self):
        """Cria mapa interativo das estações"""
        # Agregar dados por estação
        station_summary = self.df.groupby(['station_name', 'latitude', 'longitude', 'region']).agg({
            'temp_average': 'mean',
            'need_index': 'mean',
            'max_temperature': 'max',
            'min_temperature': 'min'
        }).reset_index().round(2)
        
        # Criar mapa
        fig = px.scatter_mapbox(
            station_summary,
            lat='latitude',
            lon='longitude',
            color='need_index',
            size='temp_average',
            hover_name='station_name',
            hover_data={
                'region': True,
                'temp_average': ':.1f',
                'need_index': ':.1f',
                'max_temperature': ':.1f',
                'min_temperature': ':.1f'
            },
            color_continuous_scale='RdYlBu_r',
            title='Mapa de Estações Meteorológicas - Índice de Necessidade',
            mapbox_style='open-street-map',
            zoom=3,
            center={'lat': -15, 'lon': -50}
        )
        
        fig.update_layout(height=600)
        fig.show()
    
    def generate_insights_report(self):
        """Gera relatório de insights"""
        print(f"\n💡 SOLIDÁRIA INSIGHTS - ESTRATÉGIAS INTELIGENTES")
        print("🤖 Recomendações Baseadas em Inteligência Artificial")
        print("=" * 60)
        
        insights = []
        
        # Insight 1: Regiões mais necessitadas
        region_needs = self.df.groupby('region')['need_index'].mean().sort_values(ascending=False)
        top_region = region_needs.index[0]
        insights.append(f"🎯 Região prioritária: {top_region} (índice médio: {region_needs.iloc[0]:.1f})")
        
        # Insight 2: Sazonalidade
        season_needs = self.df.groupby('season')['need_index'].mean().sort_values(ascending=False)
        peak_season = season_needs.index[0]
        insights.append(f"📅 Pico de necessidade: {peak_season} (índice: {season_needs.iloc[0]:.1f})")
        
        # Insight 3: Eventos extremos
        extreme_pct = ((self.df['need_index'] > 50).sum() / len(self.df)) * 100
        insights.append(f"🚨 {extreme_pct:.1f}% dos dias apresentam necessidade alta/crítica")
        
        # Insight 4: Variabilidade regional
        temp_variability = self.df.groupby('region')['temp_amplitude'].mean().sort_values(ascending=False)
        most_variable = temp_variability.index[0]
        insights.append(f"🌡️  Região com maior variação térmica: {most_variable} ({temp_variability.iloc[0]:.1f}°C)")
        
        # Insight 5: Correlações importantes
        temp_need_corr = self.df['temp_average'].corr(self.df['need_index'])
        if abs(temp_need_corr) > 0.3:
            direction = "positiva" if temp_need_corr > 0 else "negativa"
            insights.append(f"📊 Correlação {direction} entre temperatura e necessidade (r={temp_need_corr:.2f})")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print(f"\n🎯 RECOMENDAÇÕES ESTRATÉGICAS:")
        print(f"   • Priorizar campanhas na região {top_region}")
        print(f"   • Intensificar ações durante o {peak_season}")
        print(f"   • Manter estoque de emergência para {extreme_pct:.0f}% dos dias críticos")
        print(f"   • Adaptar estratégias para variação térmica em {most_variable}")
        
        return insights

def run_complete_eda():
    """Executa análise exploratória completa"""
    
    # Gerar dados simulados (mesmo código do sistema principal)
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
        
        # Outras regiões (sample)
        ("BELO HORIZONTE", 83587, -19.932222, -43.937778, 915.0),
        ("BRASILIA DF", 83377, -15.789444, -47.925833, 1159.5),
        ("SALVADOR", 83229, -12.910833, -38.331667, 51.4),
        ("FORTALEZA", 82397, -3.776389, -38.532222, 26.5),
        ("MANAUS", 82331, -3.102778, -60.016667, 67.0),
        ("PORTO ALEGRE", 83967, -30.053056, -51.178611, 46.97),
        ("CURITIBA", 83842, -25.448056, -49.231944, 923.5)
    ]
    
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    for date in dates[:1000]:  # Amostra para demonstração
        for station_name, station_id, lat, lng, alt in stations:
            # Simular temperaturas baseadas na região e estação
            month = date.month
            
            # Ajustar temperatura base pela latitude
            if lat > -10:  # Norte/Nordeste
                base_summer, base_winter = 35, 28
                var_summer, var_winter = 3, 2
            elif lat > -20:  # Centro-Oeste
                base_summer, base_winter = 32, 25
                var_summer, var_winter = 4, 3
            elif lat > -25:  # Sudeste
                base_summer, base_winter = 30, 22
                var_summer, var_winter = 4, 4
            else:  # Sul
                base_summer, base_winter = 28, 18
                var_summer, var_winter = 3, 5
            
            # Ajustar pela altitude
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
            if np.random.random() < 0.02:
                max_temp = np.nan
            if np.random.random() < 0.02:
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
    
    print("🤖 SOLIDÁRIA - ANÁLISE EXPLORATÓRIA INTELIGENTE")
    print("🎯 IA Descobrindo Padrões para Otimizar Doações")
    print("=" * 60)
    
    # Executar análise
    eda = SolidárIA_EDA(df)
    
    # Estatísticas gerais
    eda.overview_statistics()
    
    # Análise regional
    regional_stats = eda.regional_analysis()
    
    # Padrões sazonais
    seasonal_patterns = eda.seasonal_patterns()
    
    # Eventos extremos
    extreme_events = eda.extreme_events_analysis()
    
    # Análise de necessidade
    need_analysis = eda.need_index_analysis()
    
    # Gerar visualizações
    eda.create_visualizations()
    
    # Insights finais
    insights = eda.generate_insights_report()
    
    print(f"\n✅ SOLIDÁRIA ANÁLISE CONCLUÍDA!")
    print(f"🤖 {len(df):,} registros processados por IA de {df['station_name'].nunique()} estações")
    print(f"🎯 SolidárIA pronta para otimizar doações com inteligência climática!")

if __name__ == "__main__":
    run_complete_eda()
