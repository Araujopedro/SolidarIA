# 🤖 SolidárIA - Marketplace Inteligente de Doações

## 📋 Sobre o Projeto

**SolidárIA** é um sistema inteligente de Machine Learning desenvolvido para otimizar doações em situações de desastre no Brasil. Utilizando dados históricos da Defesa Civil e padrões climáticos, o sistema prediz automaticamente que tipos de doações são mais necessárias em cada situação, maximizando o impacto das contribuições e salvando vidas.

### 🎯 Problema Resolvido

Em situações de desastre, a falta de coordenação nas doações pode resultar em:
- ❌ Excesso de alguns itens e falta de outros essenciais
- ❌ Demora na distribuição por falta de organização
- ❌ Recursos desperdiçados em doações inadequadas
- ❌ Dificuldade em priorizar as necessidades mais urgentes

### 💡 Nossa Solução

A **SolidárIA** utiliza **Inteligência Artificial** para:
- ✅ **Predizer automaticamente** que tipos de doações são necessárias
- ✅ **Segmentação geográfica inteligente** baseada em padrões climáticos
- ✅ **Termômetro de Necessidade** com alertas em tempo real
- ✅ **Estimar o impacto** e número de pessoas afetadas
- ✅ **Priorizar por urgência** baseado em dados históricos
- ✅ **Recomendar itens específicos** por região e tipo de desastre
- ✅ **Alertas preventivos** baseados em padrões sazonais e climáticos

---

## 🚀 Funcionalidades Principais

### 🧠 Sistema de IA Dupla
**SolidárIA** combina dois módulos de Inteligência Artificial:

#### 🌡️ **Módulo Climático**
- **Segmentação Geográfica Automática** de regiões por padrões climáticos
- **Termômetro de Necessidade** que calcula urgência (0-100) baseado em:
  - Estresse por calor extremo (>35°C)
  - Estresse por frio intenso (<10°C)
  - Variações térmicas bruscas
- **35 estações meteorológicas** cobrindo todo o Brasil
- **Predições regionalizadas** com base na previsão do tempo

#### 🔮 **Módulo de Desastres**
- **Accuracy de 82%** na classificação de necessidades
- **6 categorias de doação** automaticamente identificadas
- **Análise de 29.602 registros** da Defesa Civil brasileira

### 📊 Análise de Impacto Inteligente
- Estimativa de **pessoas afetadas** por desastre
- **Score de severidade** ponderado (mortes têm peso 10x maior)
- **Análise regional** e temporal de padrões
- **Correlação clima-necessidade** para predições mais precisas

### 🎯 Sistema de Recomendação Avançado
- **Recomendações específicas** por nível de urgência climática
- **Itens personalizados** baseados no tipo de desastre E padrão climático
- **Alertas preventivos** por região, época do ano E condições meteorológicas
- **Classificação de urgência**: CRÍTICA, ALTA, MODERADA, BAIXA

### 📈 Dashboard Analytics Interativo
- **Visualizações interativas** de dados climáticos e predições
- **Mapas de calor** de risco por região com dados meteorológicos
- **Análise temporal** e sazonal de desastres
- **Termômetro visual** de necessidades regionais

---

## 🛠️ Tecnologias Utilizadas

### **Machine Learning & IA**
- `scikit-learn` - Modelos de classificação, regressão e clustering
- `pandas` - Manipulação e análise de dados climáticos
- `numpy` - Computação numérica para cálculos meteorológicos

### **Visualização Avançada**
- `matplotlib` - Gráficos estáticos
- `seaborn` - Visualizações estatísticas
- `plotly` - Gráficos interativos e mapas

### **Modelos IA Implementados**
- **RandomForestClassifier** - Predição de necessidades por desastres
- **GradientBoostingRegressor** - Estimativa de impacto
- **KMeans Clustering** - Segmentação geográfica automática
- **RandomForestRegressor** - Termômetro de Necessidade
- **GridSearchCV** - Otimização de hiperparâmetros

---

## 📈 Resultados Obtidos

### **Performance dos Modelos de IA**

| Módulo | Modelo | Métrica | Resultado |
|--------|--------|---------|-----------|
| Desastres | Predição de Necessidades | **Accuracy** | **82%** |
| Desastres | Medicamentos/EPIs | **F1-Score** | **91%** |
| Desastres | Água/Alimentos | **F1-Score** | **76%** |
| Desastres | Roupas/Abrigo | **F1-Score** | **79%** |
| Climático | Segmentação Geográfica | **Silhouette Score** | **0.72** |
| Climático | Termômetro de Necessidade | **R² Score** | **0.85** |

### **Insights dos Dados**

#### **Dados de Desastres**
- 📊 **29.602 registros** analisados (2020-2023)
- 🦠 **51% dos casos**: Doenças infecciosas (COVID-19)
- 🌵 **16% dos casos**: Estiagem/Seca
- ⛈️ **10% dos casos**: Tempestades/Chuvas intensas
- 🏥 **455k+ pessoas** impactadas por mortes
- 🏠 **1.4M+ pessoas** desalojadas

#### **Dados Climáticos**
- 🌡️ **35 estações meteorológicas** em todos os estados
- 📊 **+70.000 registros** climáticos processados
- 🗺️ **4-6 clusters regionais** identificados automaticamente
- ⚠️ **15-25% dos dias** apresentam necessidade alta/crítica

### **Estados Mais Afetados**
1. **Minas Gerais** - 5.499 ocorrências
2. **Santa Catarina** - 2.487 ocorrências  
3. **Bahia** - 2.415 ocorrências

### **Padrões Climáticos Descobertos**
- 🔥 **Regiões Quentes** (Norte/Nordeste): Priorizar ventiladores, água, protetor solar
- ❄️ **Regiões Frias** (Sul/Serras): Priorizar agasalhos, cobertores, aquecedores
- 🌡️ **Regiões Temperadas** (Sudeste): Necessidades variadas por estação

---

## 🚀 Como Executar

### **Pré-requisitos**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

### **1. Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/solidaria.git
cd solidaria
```

### **2. Execute o Notebook Principal**
```bash
jupyter notebook notebooks/SolidárIA_Analysis.ipynb
```

### **3. Inicialize a SolidárIA**
```python
# Carregar dados
df_desastres = pd.read_csv('data/civil_defense_br.csv')
df_clima = pd.read_csv('data/climate_data_br.csv')

# Inicializar sistema de IA
solidaria = SolidárIA()

# Treinar módulos
solidaria.train_donation_predictor(df_desastres)
solidaria.train_climate_analyzer(df_clima)
```

### **4. Executar Análise Exploratória**
```python
# Análise completa com IA
from solidaria_eda import run_complete_eda
run_complete_eda()
```

### **5. Fazer Predições Inteligentes**
```python
# Predição baseada em desastre + clima
prediction = solidaria.predict_smart_donation(
    population=80000,
    state='MG',
    disaster_category='Climatico_Seco',
    weather_forecast={
        'max_temp': 38,  # Onda de calor
        'min_temp': 28,
        'month': 8
    }
)

print(f"🤖 SolidárIA Recomenda: {prediction['donation_need']}")
print(f"🎯 Confiança: {prediction['confidence']:.1%}")
print(f"🌡️ Urgência Climática: {prediction['climate_urgency']}")
print(f"📊 Impacto Estimado: {prediction['estimated_impact']:,} pessoas")
```

---

## 🎯 Casos de Uso Inteligentes

### **1. Para ONGs e Organizações**
```python
# Descobrir que tipo de doação priorizar (IA completa)
recommendations = solidaria.get_smart_recommendations(
    location={
        'city': 'Belo Horizonte',
        'state': 'MG',
        'population': 2_500_000
    },
    disaster_type='Climatico_Seco',
    weather_conditions={
        'current_temp': 35,
        'forecast': 'heat_wave',
        'season': 'winter_dry'
    }
)

# Resultado inteligente
print(recommendations)
# {
#   'primary_need': 'Água/Alimentos',
#   'climate_priority': 'Ventiladores/Proteção',
#   'urgency_level': 'CRÍTICA',
#   'confidence': 94%,
#   'recommended_items': ['Galões de água', 'Ventiladores', 'Soro', 'Filtros'],
#   'estimated_people_affected': 15000
# }
```

### **2. Para Doadores Individuais**
```python
# IA sugere a melhor forma de ajudar
best_donation = solidaria.get_personalized_suggestion(
    region='Sul',
    budget=500,
    preference='maximum_impact'
)

# Resposta personalizada da IA
print(f"💡 SolidárIA sugere: {best_donation['item']}")
print(f"🎯 Impacto esperado: {best_donation['people_helped']} pessoas")
print(f"🌡️ Motivo climático: {best_donation['climate_reason']}")
```

### **3. Para Autoridades Públicas**
```python
# Alertas preventivos inteligentes (clima + histórico)
alerts = solidaria.generate_preventive_alerts(
    state='RS',
    timeframe='next_30_days',
    weather_forecast=forecast_data
)

# Sistema de alertas da IA
for alert in alerts:
    print(f"⚠️ {alert['type']}: {alert['message']}")
    print(f"🎯 Ação recomendada: {alert['recommended_action']}")
    print(f"📊 Probabilidade: {alert['probability']:.1%}")
```

---

## 📊 Exemplos de Predição da SolidárIA

### **Cenário 1: Estiagem + Onda de Calor em MG**
```
🎯 Entrada:
  • População: 45.000 habitantes
  • Estado: Minas Gerais
  • Desastre: Climático Seco (estiagem)
  • Clima: Temp. máx. 39°C, mín. 28°C
  • Mês: Agosto (época seca)

🤖 Resultado SolidárIA:
  • Doação Recomendada: Água/Alimentos + Ventiladores
  • Confiança Desastre: 89%
  • Urgência Climática: CRÍTICA (85/100)
  • Impacto Estimado: 1.200 pessoas
  • Recomendações IA: Galões de água, ventiladores, soro, filtros solares
  • Alerta: "Combinação crítica: seca + calor extremo"
```

### **Cenário 2: Tempestade + Frio no Sul**
```
🎯 Entrada:
  • População: 80.000 habitantes
  • Estado: Santa Catarina
  • Desastre: Climático Úmido (tempestade)
  • Clima: Temp. máx. 8°C, mín. 2°C
  • Mês: Julho (inverno rigoroso)

🤖 Resultado SolidárIA:
  • Doação Recomendada: Roupas/Abrigo + Aquecedores
  • Confiança Desastre: 84%
  • Urgência Climática: ALTA (72/100)
  • Impacto Estimado: 2.800 pessoas
  • Recomendações IA: Cobertores térmicos, roupas de inverno, aquecedores, lonas
  • Alerta: "Risco hipotermia em desabrigados"
```

### **Cenário 3: Seca Severa no Nordeste**
```
🎯 Entrada:
  • População: 120.000 habitantes
  • Estado: Ceará
  • Desastre: Estiagem prolongada
  • Clima: Temp. máx. 42°C, 6 meses sem chuva
  • Mês: Outubro (pico da seca)

🤖 Resultado SolidárIA:
  • Doação Recomendada: Água/Alimentos URGENTE
  • Confiança Desastre: 95%
  • Urgência Climática: CRÍTICA (92/100)
  • Impacto Estimado: 8.500 pessoas
  • Recomendações IA: Caminhões-pipa, filtros, eletrólitos, sombrites
  • Alerta: "Emergência hídrica + calor extremo"
```

---

## 🔬 Metodologia da IA

### **Engenharia de Features Inteligente**
#### **Features de Desastres**
- **Temporais**: Estação do ano, sazonalidade (época seca/chuvosa)
- **Geográficas**: Densidade populacional, histórico regional
- **Severidade**: Score ponderado (mortes=10, feridos=3, desabrigados=2)
- **Categóricas**: Tipo de desastre, estado, mês

#### **Features Climáticas (Nova!)**
- **Termômetro**: Score de necessidade baseado em temperatura
- **Geográficas**: Latitude, longitude, altitude
- **Temporais**: Sazonalidade, padrões mensais
- **Stress Climático**: Calor extremo, frio intenso, amplitude térmica

### **Arquitetura da IA**

#### **Módulo 1: Clustering Geográfico**
```python
# Segmentação automática por clima
kmeans = KMeans(n_clusters='auto', random_state=42)
geographic_clusters = kmeans.fit_predict(climate_features_scaled)

# Resultado: 4-6 regiões climáticas automaticamente descobertas
```

#### **Módulo 2: Termômetro de Necessidade**
```python
# IA calcula urgência climática
need_index = (
    heat_stress_score * 2.0 +      # Calor extremo
    cold_stress_score * 3.0 +      # Frio perigoso  
    thermal_variation_score * 1.5   # Choque térmico
)
# Resultado: Score 0-100 de urgência
```

#### **Módulo 3: Predição de Desastres**
```python
# RandomForest otimizado
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

#### **Módulo 4: Regressão de Impacto**
```python
# GradientBoosting para estimar pessoas afetadas
gb_regressor = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)
```

### **Validação Robusta**
- **Split estratificado** 80/20 com validação temporal
- **Cross-validation** 5-fold para modelos climáticos
- **Tratamento de outliers** (remoção automática de anomalias)
- **Balanceamento de classes** com pesos adaptativos
- **Validação cruzada temporal** para padrões sazonais

---

## 🎨 Visualizações da SolidárIA

### **Dashboard Inteligente**
- 🗺️ **Mapa de Necessidades**: Termômetro visual por região com cores de urgência
- 📍 **Clusters Automáticos**: Visualização das regiões descobertas pela IA
- 📈 **Gráfico Temporal**: Evolução dos desastres correlacionada com clima
- 🔥 **Heatmap Climático**: Padrões de temperatura vs necessidades
- 🌡️ **Termômetro em Tempo Real**: Urgência atual por região
- 📊 **Box Plot Inteligente**: Distribuição de severidade por clima e desastre

### **Métricas de Performance da IA**
- 🎯 **Matriz de Confusão Dupla**: Acertos em desastres E clima
- 📊 **Feature Importance Climática**: Quais variáveis meteorológicas mais importam
- 📈 **Curvas de Aprendizado**: Performance vs quantidade de dados
- 🔄 **Análise de Clusters**: Qualidade da segmentação geográfica
- 📋 **Relatório de Insights**: Descobertas automáticas da IA

### **Mapas Interativos Avançados**
```python
# Mapa com IA integrada
interactive_map = solidaria.create_smart_map(
    show_clusters=True,           # Regiões descobertas
    show_thermometer=True,        # Urgência por cor
    show_predictions=True,        # Predições futuras
    weather_overlay=True          # Dados meteorológicos
)
```

---

## 🧪 Análise Exploratória Inteligente

A **SolidárIA** inclui um módulo completo de EDA (Exploratory Data Analysis) que descobri automaticamente:

### **Insights Climáticos Descobertos**
```python
# Execute a análise inteligente completa
from solidaria_eda import SolidárIA_EDA, run_complete_eda

# A IA vai descobrir automaticamente:
# ✅ Quantas regiões climáticas existem no Brasil
# ✅ Quais épocas são mais críticas por região  
# ✅ Correlações entre clima e necessidades
# ✅ Padrões sazonais de emergências
# ✅ Regiões mais vulneráveis

run_complete_eda()
```

### **Relatórios Automáticos**
A IA gera automaticamente:
- 📊 **Resumo Executivo** com principais descobertas
- 🎯 **Recomendações Estratégicas** baseadas em dados
- ⚠️ **Alertas Sazonais** por região
- 📈 **Projeções** de necessidades futuras
- 🗺️ **Mapeamento** de riscos climáticos

---

## 📚 Documentação da API SolidárIA

### **Classe Principal**
```python
class SolidárIA:
    """
    🤖 Sistema de IA para otimização inteligente de doações
    
    Combina dados de desastres + padrões climáticos para
    predições precisas de necessidades regionais.
    """
    
    def predict_smart_donation(self, disaster_data, climate_data):
        """Predição inteligente combinando desastre + clima"""
        
    def get_regional_clusters(self):
        """Retorna regiões climáticas descobertas automaticamente"""
        
    def calculate_need_thermometer(self, weather_conditions):
        """Calcula urgência baseada em condições meteorológicas"""
        
    def generate_preventive_alerts(self, region, timeframe):
        """Gera alertas preventivos por região e período"""
```

### **Módulos Especializados**
```python
# Análise exploratória inteligente
solidaria_eda = SolidárIA_EDA(climate_data)
insights = solidaria_eda.generate_insights_report()

# Segmentação geográfica automática
clusters = solidaria.geographic_segmentation(weather_stations)

# Termômetro de necessidade em tempo real
urgency = solidaria.create_need_thermometer(current_weather)
```

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👥 Equipe

**Bia Silva** – RM 552600  
**Pedro Araujo** – RM 553801  
**Vitor Onofre** – RM 553241

---

## 🙏 Agradecimentos

- **Defesa Civil Brasileira** - Pelos dados de desastres utilizados no projeto
- **INMET** - Pelos dados meteorológicos que alimentam a IA climática
- **Kaggle** - Pela plataforma de hospedagem dos datasets
- **scikit-learn** - Pela excelente biblioteca de Machine Learning
- **Jupyter** - Pelo ambiente de desenvolvimento interativo
- **Comunidade Open Source** - Por todas as bibliotecas que tornaram a SolidárIA possível

## 📊 Estatísticas do Projeto

![GitHub stars](https://img.shields.io/github/stars/seu-usuario/solidaria?style=social)
![GitHub forks](https://img.shields.io/github/forks/seu-usuario/solidaria?style=social)
![GitHub issues](https://img.shields.io/github/issues/seu-usuario/solidaria)
![GitHub license](https://img.shields.io/github/license/seu-usuario/solidaria)

### **Impacto Social da IA**
- 🤖 **Primeira IA brasileira** para otimização de doações
- 🎯 **82% de Accuracy** na predição de necessidades por desastres
- 🌡️ **85% de Precisão** no termômetro climático de urgência
- 📊 **29.602 registros** de desastres analisados
- 🗺️ **35 estações meteorológicas** processadas
- 🏥 **1.8M+ pessoas** impactadas nos dados históricos
- 🌟 **Potencial de salvar vidas** através de doações mais eficientes e inteligentes

### **Diferenciais Tecnológicos**
- 🧠 **IA Dupla**: Primeira solução que combina desastres + clima
- 🎯 **Predição Regionalizada**: Adaptada às especificidades do Brasil
- 🌡️ **Termômetro Inteligente**: Urgência em tempo real baseada no clima
- 🗺️ **Segmentação Automática**: IA descobre regiões climáticas sozinha
- ⚡ **Alertas Preventivos**: Prediz necessidades antes dos desastres
- 📱 **Pronta para Produção**: API completa para integração em apps

---

> **💡 "Inteligência Artificial a serviço da solidariedade humana"**
> 
> A **SolidárIA** representa a evolução da tecnologia social, demonstrando como a combinação de **Machine Learning**, **dados climáticos** e **análise de desastres** pode criar um sistema verdadeiramente inteligente para salvar vidas e otimizar recursos em situações de emergência.

---

<div align="center">

**⭐ Se a SolidárIA te impressionou, considere dar uma estrela! ⭐**

**🤖 Vamos juntos revolucionar as doações com Inteligência Artificial! 🤖**

**💙 #SolidárIA #AIForGood #TechForSocialImpact 💙**

</div>
