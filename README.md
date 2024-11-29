# Sistema de Análise de Dados e Predição com Machine Learning

Este é um projeto de aplicação web que permite a análise de dados e predições utilizando Machine Learning. 
O sistema suporta upload de datasets no formato CSV, oferece diversas visualizações interativas e permite o treinamento dinâmico de modelos com ajuste de hiperparâmetros.

---

## Funcionalidades
- Upload de arquivos CSV para análise de dados.
- Geração automática de gráficos estáticos e interativos, incluindo:
  - Gráficos de barras, pizza e histogramas.
  - Mapas interativos para visualização de dados geográficos.
- Treinamento dinâmico de modelos de machine learning:
  - **Modelos suportados**: Random Forest, Decision Tree, XGBoost, Logistic Regression.
  - Ajuste de hiperparâmetros diretamente na interface.
- Geração de relatórios detalhados em PDF contendo gráficos, métricas e insights do modelo.
- Interface amigável para predições personalizadas com base em atributos fornecidos pelo usuário.

---

## **Requisitos**
- **Linguagem**: Python 3.8 ou superior.
- **Dependências**:
  - Flask
  - pandas
  - scikit-learn
  - matplotlib
  - plotly
  - xgboost
  - reportlab
- **Ferramentas adicionais**:
  - Navegador atualizado (Google Chrome, Firefox).

---
## **Instalação**
1. **Clone o repositório**:
   ```bash
   git clone https://github.com/rafael-deangelo/ev-analysis-app
   cd ev-analysis-app

2. **Crie um ambiente virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt

4. **Execute a aplicação**:
   ```bash
   python app.py

5. **Acesse no navegador**:
   ```bash
   URL padrão: http://127.0.0.1:5000

---

## **Como Usar**
### **1. Carregar um Dataset**
- Faça o upload de um arquivo CSV na página inicial.
- Certifique-se de que o arquivo contém as colunas esperadas:
  - `Model Year`, `Electric Range`, `Base MSRP`, `Make`, `Model`, `Clean Alternative Fuel Vehicle (CAFV) Eligibility`, `Electric Vehicle Type`.

### **2. Visualizar os Dados**
- Após o upload, gráficos e análises serão gerados automaticamente, incluindo:
  - Distribuição dos fabricantes mais comuns.
  - Tipos de veículos elétricos.
  - Distribuição do alcance elétrico.

### **3. Treinar o Modelo**
- O sistema treinará o modelo usando os hiperparâmetros padrão.
- Você pode ajustar os hiperparâmetros na tela de resultados:
  - **`n_estimators`**: Número de árvores no Random Forest ou XGBoost.
  - **`max_depth`**: Profundidade máxima das árvores.

### **4. Fazer Predições**
- Insira os valores desejados para variáveis como:
  - Ano de fabricação, alcance elétrico, preço base, fabricante e modelo.
- O sistema retornará a predição do tipo de veículo elétrico, como:
  - **Exemplo de Resultado**: "Battery Electric Vehicle".

### **5. Gerar Relatórios**
- Na tela de resultados, clique para baixar relatórios detalhados em PDF.
- Os relatórios incluem gráficos, métricas de avaliação do modelo e insights sobre o dataset.

---

## **Arquitetura do Código**
- **`app.py`**:
  - Arquivo principal que implementa a aplicação Flask.
- **`templates/`**:
  - Contém os arquivos HTML usados para renderizar a interface do usuário.
- **`static/`**:
  - Arquivos estáticos, como gráficos, estilos CSS e mapas interativos.
- **`requirements.txt`**:
  - Lista de dependências do projeto.
- **`Electric_Vehicle_Population_Data.csv`**:
  - Dataset usado como exemplo para a análise.

---

## **Exemplo de Uso**
- **Configuração Inicial**:
  - Faça o upload do dataset fornecido ou de um dataset no mesmo formato.
- **Visualização dos Dados**:
  - Após o upload, você verá gráficos detalhados sobre o dataset, como:
    - Gráfico de barras dos fabricantes mais comuns.
    - Gráfico de pizza dos tipos de veículos elétricos.
    - Histograma da distribuição do alcance elétrico.
- **Predição**:
  - **Exemplo de Entrada**:
    - Ano do Modelo: `2023`
    - Alcance Elétrico: `250` milhas
    - Preço Base: `$35,000`
    - Fabricante: `Tesla`
    - Modelo: `Model 3`
  - **Resultado Esperado**:
    - Tipo de Veículo: **"Battery Electric Vehicle"**.

---

## **Agradecimentos**
Este projeto utiliza bibliotecas e frameworks open-source, incluindo:
- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [Plotly](https://plotly.com/)
- [pandas](https://pandas.pydata.org/)

---