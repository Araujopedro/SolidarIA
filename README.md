# 🤖 DoaciAI - Marketplace Inteligente de Doações

## 📋 Sobre o Projeto

**DoaciAI** é um sistema inteligente de Machine Learning desenvolvido para otimizar doações em situações de desastre no Brasil. Utilizando dados históricos da Defesa Civil, o sistema prediz automaticamente que tipos de doações são mais necessárias em cada situação, maximizando o impacto das contribuições e salvando vidas.

### 🎯 Problema Resolvido

Em situações de desastre, a falta de coordenação nas doações pode resultar em:
- ❌ Excesso de alguns itens e falta de outros essenciais
- ❌ Demora na distribuição por falta de organização
- ❌ Recursos desperdiçados em doações inadequadas
- ❌ Dificuldade em priorizar as necessidades mais urgentes

### 💡 Nossa Solução

O DoaciAI utiliza **Machine Learning** para:
- ✅ **Predizer automaticamente** que tipos de doações são necessárias
- ✅ **Estimar o impacto** e número de pessoas afetadas
- ✅ **Priorizar por urgência** baseado em dados históricos
- ✅ **Recomendar itens específicos** por região e tipo de desastre
- ✅ **Alertas preventivos** baseados em padrões sazonais

---

## 🚀 Funcionalidades Principais

### 🔮 Predição Inteligente
- **Accuracy de 82%** na classificação de necessidades
- **6 categorias de doação** automaticamente identificadas
- **Análise de 29.602 registros** da Defesa Civil brasileira

### 📊 Análise de Impacto
- Estimativa de **pessoas afetadas** por desastre
- **Score de severidade** ponderado (mortes têm peso 10x maior)
- **Análise regional** e temporal de padrões

### 🎯 Sistema de Recomendação
- **Recomendações específicas** por nível de urgência
- **Itens personalizados** baseados no tipo de desastre
- **Alertas preventivos** por região e época do ano

### 📈 Dashboard Analytics
- **Visualizações interativas** de dados e predições
- **Mapas de calor** de risco por região
- **Análise temporal** e sazonal de desastres

---

## 🛠️ Tecnologias Utilizadas

### **Machine Learning**
- `scikit-learn` - Modelos de classificação e regressão
- `pandas` - Manipulação e análise de dados
- `numpy` - Computação numérica

### **Visualização**
- `matplotlib` - Gráficos estáticos
- `seaborn` - Visualizações estatísticas
- `plotly` - Gráficos interativos

### **Modelos Implementados**
- **RandomForestClassifier** - Predição de necessidades
- **GradientBoostingRegressor** - Estimativa de impacto
- **GridSearchCV** - Otimização de hiperparâmetros

---

## 📈 Resultados Obtidos

### **Performance dos Modelos**

| Modelo | Métrica | Resultado |
|--------|---------|-----------|
| Predição de Necessidades | **Accuracy** | **82%** |
| Medicamentos/EPIs | **F1-Score** | **91%** |
| Água/Alimentos | **F1-Score** | **76%** |
| Roupas/Abrigo | **F1-Score** | **79%** |

### **Insights dos Dados**

- 📊 **29.602 registros** analisados (2020-2023)
- 🦠 **51% dos casos**: Doenças infecciosas (COVID-19)
- 🌵 **16% dos casos**: Estiagem/Seca
- ⛈️ **10% dos casos**: Tempestades/Chuvas intensas
- 🏥 **455k+ pessoas** impactadas por mortes
- 🏠 **1.4M+ pessoas** desalojadas

### **Estados Mais Afetados**
1. **Minas Gerais** - 5.499 ocorrências
2. **Santa Catarina** - 2.487 ocorrências  
3. **Bahia** - 2.415 ocorrências


```

---

## 🚀 Como Executar

### **Pré-requisitos**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

### **1. Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/doaciai.git
cd doaciai
```

### **2. Execute o Notebook**
```bash
jupyter notebook notebooks/DoaciAI_Analysis.ipynb
```

### **3. Treine os Modelos**
```python
# Carregar dados
df = pd.read_csv('data/civil_defense_br.csv')

# Inicializar sistema
ml_system = DonationMLSystem()

# Treinar modelos
ml_system.train_donation_predictor(df_processed)
ml_system.train_impact_predictor(df_processed)
```

### **4. Fazer Predições**
```python
# Exemplo de predição
prediction = ml_system.predict(
    population=80000,
    state='MG',
    disaster_category='Climatico_Seco',
    month=8,  # Agosto - época seca
    severity_score=200
)

print(f"Doação recomendada: {prediction['donation_need']}")
print(f"Confiança: {prediction['confidence']:.1%}")
print(f"Impacto estimado: {prediction['estimated_impact']:,} pessoas")
```

---

## 🎯 Casos de Uso

### **1. Para ONGs e Organizações**
```python
# Descobrir que tipo de doação priorizar
recommendations = ml_system.predict(
    population=cidade['populacao'],
    state=cidade['estado'],
    disaster_category=tipo_desastre,
    month=mes_atual
)
```

### **2. Para Doadores Individuais**
```python
# Encontrar a melhor forma de ajudar
best_donation = ml_system.get_recommendations(
    donation_type='Medicamentos_EPIs',
    impact_level=500  # pessoas afetadas
)
```

### **3. Para Autoridades Públicas**
```python
# Alertas preventivos por região
alerts = create_preventive_alerts(
    state='RS',
    month=12  # Dezembro - época de chuvas no Sul
)
```

---

## 📊 Exemplos de Predição

### **Cenário 1: Estiagem em MG**
```
🎯 Entrada:
  • População: 45.000 habitantes
  • Estado: Minas Gerais
  • Tipo: Climático Seco (estiagem)
  • Mês: Agosto (época seca)

📊 Resultado:
  • Doação Recomendada: Água/Alimentos
  • Confiança: 89%
  • Impacto Estimado: 1.200 pessoas
  • Recomendações: Filtros de água, cestas básicas, galões
```

### **Cenário 2: Tempestade no Sul**
```
🎯 Entrada:
  • População: 80.000 habitantes
  • Estado: Santa Catarina
  • Tipo: Climático Úmido (tempestade)
  • Mês: Dezembro (verão chuvoso)

📊 Resultado:
  • Doação Recomendada: Roupas/Abrigo
  • Confiança: 84%
  • Impacto Estimado: 2.800 pessoas
  • Recomendações: Cobertores, roupas, lonas
```

---

## 🔬 Metodologia

### **Engenharia de Features**
- **Temporais**: Estação do ano, sazonalidade (época seca/chuvosa)
- **Geográficas**: Densidade populacional, histórico regional
- **Severidade**: Score ponderado (mortes=10, feridos=3, desabrigados=2)
- **Categóricas**: Tipo de desastre, estado, mês

### **Modelos Utilizados**

#### **RandomForestClassifier**
```python
# Otimização com GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'class_weight': 'balanced'
}
```

#### **GradientBoostingRegressor**
```python
# Para estimativa de impacto
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7]
}
```

### **Validação**
- **Split estratificado** 80/20
- **Cross-validation** 3-fold
- **Tratamento de outliers** (remoção do percentil 99)
- **Balanceamento de classes**




```

---

## 🎨 Visualizações

### **Dashboard Interativo**
- 📍 **Mapa de Calor**: Distribuição de desastres por estado
- 📈 **Gráfico Temporal**: Evolução dos desastres ao longo do tempo  
- 🔥 **Heatmap Sazonal**: Padrões por mês e tipo de desastre
- 📊 **Box Plot**: Distribuição de severidade por categoria

### **Métricas de Performance**
- 🎯 **Matriz de Confusão**: Visualização dos acertos/erros
- 📊 **Feature Importance**: Quais variáveis mais influenciam as predições
- 📈 **Curvas de Aprendizado**: Performance vs quantidade de dados



## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👥 Equipe

Bia silva – Rm 552600
Pedro Araujo – Rm553801
Vitor Onofre – Rm553241



## 🙏 Agradecimentos

- **Defesa Civil Brasileira** - Pelos dados utilizados no projeto
- **Kaggle** - Pela plataforma de hospedagem do dataset
- **scikit-learn** - Pela excelente biblioteca de Machine Learning
- **Jupyter** - Pelo ambiente de desenvolvimento interativo
- **Comunidade Open Source** - Por todas as bibliotecas utilizadas

## 📊 Estatísticas do Projeto

![GitHub stars](https://img.shields.io/github/stars/seu-usuario/doaciai?style=social)
![GitHub forks](https://img.shields.io/github/forks/seu-usuario/doaciai?style=social)
![GitHub issues](https://img.shields.io/github/issues/seu-usuario/doaciai)
![GitHub license](https://img.shields.io/github/license/seu-usuario/doaciai)

### **Impacto Social**
- 🎯 **82% de Accuracy** na predição de necessidades
- 📊 **29.602 registros** analisados
- 🏥 **1.8M+ pessoas** impactadas nos dados
- 🌟 **Potencial de salvar vidas** através de doações mais eficientes

---

> **💡 "A tecnologia a serviço da solidariedade humana"**
> 
> O DoaciAI representa a união entre **Inteligência Artificial** e **impacto social**, demonstrando como Machine Learning pode ser usado para resolver problemas reais e salvar vidas em situações de emergência.

---

<div align="center">

**⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐**

**🤝 Vamos juntos tornar as doações mais inteligentes e eficazes! 🤝**

</div>
