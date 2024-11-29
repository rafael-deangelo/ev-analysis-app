from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file
from reportlab.lib import colors

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Substituir por uma chave mais forte em produção

trained_model = None  # Variável para o modelo treinado
label_encoder = None  # Variável para o codificador de labels

# Lista global de classificadores suportados
SUPPORTED_CLASSIFIERS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
}

# Variável global para armazenar o classificador selecionado
selected_classifier = "random_forest"  # Classificador padrão

def generate_bar_chart(df):
    """Gera um gráfico de barras para os fabricantes mais comuns."""
    top_makes = df['Make'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_makes.plot(kind='bar', color='skyblue')
    plt.title("Top 10 Fabricantes de Veículos Elétricos")
    plt.xlabel("Fabricante")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/top_makes.png')
    plt.close()

def generate_pie_chart(df):
    """Gera um gráfico de pizza para os tipos de veículos elétricos."""
    vehicle_types = df['Electric Vehicle Type'].value_counts()
    plt.figure(figsize=(8, 8))
    vehicle_types.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['gold', 'lightblue', 'lightgreen'])
    plt.title("Distribuição de Tipos de Veículos Elétricos")
    plt.tight_layout()
    plt.savefig('static/vehicle_types.png')
    plt.close()

def generate_range_histogram(df):
    """Gera um histograma para distribuição de alcance elétrico."""
    plt.figure(figsize=(10, 6))
    df['Electric Range'].plot(kind='hist', bins=20, color='lightblue', edgecolor='black')
    plt.title("Distribuição do Alcance Elétrico")
    plt.xlabel("Alcance Elétrico (milhas)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig('static/range_histogram.png')
    plt.close()

def generate_model_year_bar_chart(df):
    """Gera um gráfico de barras para veículos por ano do modelo."""
    model_year_counts = df['Model Year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    model_year_counts.plot(kind='bar', color='coral', edgecolor='black')
    plt.title("Quantidade de Veículos por Ano do Modelo")
    plt.xlabel("Ano do Modelo")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig('static/model_year_bar_chart.png')
    plt.close()

def generate_avg_range_by_make(df):
    """Gera um gráfico de barras para alcance médio por fabricante."""
    avg_range = df.groupby('Make')['Electric Range'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    avg_range.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Alcance Médio por Fabricante (Top 10)")
    plt.xlabel("Fabricante")
    plt.ylabel("Alcance Médio (milhas)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/avg_range_by_make.png')
    plt.close()

def generate_interactive_bar_chart(df):
    """Gera um gráfico de barras interativo para os fabricantes mais comuns."""
    top_makes = df['Make'].value_counts().head(10).reset_index()
    top_makes.columns = ['Make', 'Count']
    fig = px.bar(top_makes, x='Make', y='Count', title="Top 10 Fabricantes de Veículos Elétricos")
    fig.update_layout(xaxis_title="Fabricante", yaxis_title="Quantidade")
    file_path = os.path.join('static', 'interactive_bar_chart.html')
    fig.write_html(file_path)

def generate_interactive_pie_chart(df):
    vehicle_types = df['Electric Vehicle Type'].value_counts().reset_index()
    vehicle_types.columns = ['Type', 'Count']
    fig = px.pie(vehicle_types, names='Type', values='Count', title="Distribuição de Tipos de Veículos Elétricos")
    file_path = os.path.join('static', 'interactive_pie_chart.html')
    fig.write_html(file_path)

def generate_map_by_region(df, region_col="State", value_col="Electric Vehicle Type"):
    """
    Gera um mapa interativo por região.
    Args:
        df (DataFrame): Dataset contendo os dados.
        region_col (str): Nome da coluna de região (ex: 'City', 'State').
        value_col (str): Coluna para agregação (ex: 'Electric Vehicle Type').
    Returns:
        None. Salva o mapa em arquivo HTML.
    """
    # Agregar dados por região
    region_data = df.groupby(region_col)[value_col].count().reset_index()
    region_data.columns = [region_col, "Count"]

    # Gerar o mapa
    fig = px.choropleth(
        region_data,
        locations=region_col,
        locationmode="USA-states",  # ou "country names" para países
        color="Count",
        hover_name=region_col,
        title=f"Densidade de Veículos Elétricos por {region_col}",
        color_continuous_scale="Viridis",
        scope="usa"  # Para restringir aos EUA; remova para mapa global
    )

    # Salvar como arquivo HTML
    map_file = "static/map_by_region.html"
    fig.write_html(map_file)
    print(f"Mapa salvo em: {map_file}")

def generate_map_by_city(df, state_col, city_col, value_col, state_filter):
    """
    Gera um mapa interativo para as cidades dentro de um estado específico.
    Args:
        df (DataFrame): Dataset contendo os dados.
        state_col (str): Nome da coluna de estado.
        city_col (str): Nome da coluna de cidade.
        value_col (str): Coluna para agregação.
        state_filter (str): Nome do estado para filtrar.
    """
    # Filtrar os dados para o estado selecionado
    filtered_data = df[df[state_col] == state_filter]

    # Verificar se há dados para o estado
    if filtered_data.empty:
        print(f"Nenhum dado encontrado para o estado: {state_filter}")
        return

    # Agregar dados por cidade
    city_data = filtered_data.groupby(city_col)[value_col].count().reset_index()
    city_data.columns = [city_col, "Count"]

    # Gerar o gráfico
    fig = px.bar(
        city_data,
        x=city_col,
        y="Count",
        title=f"Distribuição de Veículos Elétricos nas Cidades de {state_filter}",
        labels={city_col: "Cidade", "Count": "Quantidade"},
        color="Count",
        color_continuous_scale="Viridis"
    )

    # Salvar como arquivo HTML
    map_file = f"static/map_by_city_{state_filter}.html"
    fig.write_html(map_file)
    print(f"Mapa por cidade salvo em: {os.path.abspath(map_file)}")

def generate_all_visualizations(df):
    """Gera todos os gráficos para análise."""
    generate_bar_chart(df)
    generate_pie_chart(df)
    generate_range_histogram(df)
    generate_model_year_bar_chart(df)
    generate_avg_range_by_make(df)
    generate_interactive_bar_chart(df)
    generate_interactive_pie_chart(df)

def generate_pdf_report(file_path, accuracy, conf_matrix):
    pdf_file_path = os.path.join("static", "analysis_report.pdf")
    try:
        print(f"Iniciando a geração do PDF para: {file_path}")
        print(f"PDF será salvo em: {pdf_file_path}")

        # Início da geração do PDF
        c = canvas.Canvas(pdf_file_path, pagesize=letter)

        # Cabeçalho
        c.setFont("Helvetica-Bold", 16)
        c.drawString(30, 770, "Relatório de Análise de Dados")
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.grey)
        c.drawString(30, 750, "Gerado pela Aplicação de Análise de Veículos Elétricos")
        c.setFillColor(colors.black)
        c.line(30, 740, 580, 740)  # Linha divisória

        # Informações gerais
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, 720, f"Arquivo Analisado:")
        c.setFont("Helvetica", 12)
        c.drawString(150, 720, file_path)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, 700, f"Acurácia do Modelo:")
        c.setFont("Helvetica", 12)
        c.drawString(150, 700, f"{accuracy:.2f}")

        # Matriz de Confusão
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, 680, "Matriz de Confusão:")
        c.setFont("Helvetica", 12)
        y_offset = 660
        for row in conf_matrix:
            c.drawString(50, y_offset, f"{row}")
            y_offset -= 20

        # Adicionar gráficos com título
        y_offset -= 30
        graphics = [
            ("static/top_makes.png", "Top 10 Fabricantes de Veículos Elétricos"),
            ("static/vehicle_types.png", "Distribuição de Tipos de Veículos Elétricos"),
            ("static/range_histogram.png", "Distribuição do Alcance Elétrico"),
            ("static/model_year_bar_chart.png", "Quantidade de Veículos por Ano do Modelo"),
            ("static/avg_range_by_make.png", "Alcance Médio por Fabricante (Top 10)")
        ]

        for graph_path, graph_title in graphics:
            if os.path.exists(graph_path):
                if y_offset < 200:  # Criar nova página se o espaço acabar
                    c.showPage()
                    y_offset = 750

                c.setFont("Helvetica-Bold", 12)
                c.drawString(30, y_offset, graph_title)
                c.drawImage(graph_path, 30, y_offset - 120, width=500, height=150)
                y_offset -= 180
            else:
                print(f"Gráfico {graph_path} não encontrado.")

        # Rodapé
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.grey)
        c.drawString(30, 50, "Relatório gerado automaticamente pela aplicação de análise de veículos elétricos.")
        c.setFillColor(colors.black)

        # Finalizar PDF
        c.showPage()
        c.save()

        print(f"PDF criado com sucesso em: {pdf_file_path}")
        return pdf_file_path
    except Exception as e:
        print(f"Erro ao gerar o PDF: {str(e)}")
        raise

def train_model(df, hyperparameters=None):
    """Treina um modelo para prever o tipo de veículo elétrico com hiperparâmetros customizáveis."""
    # Seleção de colunas relevantes
    features = ['Model Year', 'Electric Range', 'Base MSRP', 'Make', 'Model',
                'Clean Alternative Fuel Vehicle (CAFV) Eligibility']
    target = 'Electric Vehicle Type'

    # Remover valores nulos
    df = df[features + [target]].dropna()

    # Codificar variáveis categóricas e o alvo
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    # Transformar variáveis categóricas em dummies
    df = pd.get_dummies(df, columns=['Make', 'Model', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'])

    # Dividir os dados em treino e teste
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar hiperparâmetros
    if hyperparameters is None:
        hyperparameters = {}  # Hiperparâmetros padrão

    # Certificar-se de que random_state não está duplicado
    hyperparameters['random_state'] = 42

    # Treinar o modelo
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Imprimir relatório no console (opcional)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"Acurácia: {accuracy}")

    # Retornar o modelo treinado, codificador e métricas
    return model, label_encoder, accuracy, conf_matrix

def calculate_metrics(y_test, y_pred, label_encoder):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy

# Lista de colunas obrigatórias no dataset
REQUIRED_COLUMNS = [
    "Model Year", "Electric Range", "Base MSRP", "Make",
    "Electric Vehicle Type", "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
]

# Rota principal para upload
@app.route("/", methods=["GET", "POST"])
def index():
    classifier_name = selected_classifier.replace("_", " ").capitalize()

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            file_path = f"./{file.filename}"
            file.save(file_path)

            try:
                df = pd.read_csv(file_path)
                missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_columns:
                    flash(f"O dataset está faltando as colunas: {', '.join(missing_columns)}")
                    return redirect(url_for("index"))
            except Exception as e:
                flash(f"Erro ao processar o arquivo: {str(e)}")
                return redirect(url_for("index"))

            return redirect(url_for("analyze", file_path=file.filename))
        else:
            flash("Por favor, envie um arquivo CSV válido.")
    return render_template("index.html", classifier_name=selected_classifier, supported_classifiers=SUPPORTED_CLASSIFIERS)

@app.route("/select_classifier", methods=["POST"])
def select_classifier():
    global selected_classifier
    selected_classifier = request.form["classifier"]
    flash(f"Classificador configurado para: {selected_classifier.capitalize()}")
    return redirect(url_for("index"))

@app.route("/set_hyperparameters", methods=["GET", "POST"])
def set_hyperparameters():
    if request.method == "POST":
        # Coletar hiperparâmetros do formulário
        n_estimators = int(request.form.get("n_estimators", 100))
        max_depth = request.form.get("max_depth")
        max_depth = int(max_depth) if max_depth else None
        min_samples_split = int(request.form.get("min_samples_split", 2))
        min_samples_leaf = int(request.form.get("min_samples_leaf", 1))

        # Salvar os hiperparâmetros como variáveis globais
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        flash("Hiperparâmetros configurados com sucesso!")
        return redirect(url_for("index"))

    # Renderizar o formulário de configuração de hiperparâmetros
    return render_template("set_hyperparameters.html")

#Nova Rota para Tratar Valores Nulos
@app.route("/handle_nulls/<file_path>", methods=["POST"])
def handle_nulls(file_path):
    try:
        # Carregar o dataset
        df = pd.read_csv(file_path)

        # Decisão do usuário sobre valores nulos
        action = request.form.get("action")
        if action == "remove":
            df = df.dropna()  # Remove linhas com valores nulos
            flash("As linhas com valores nulos foram removidas.")
        elif action == "fill":
            # Preencher valores nulos
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())  # Preenche valores numéricos com a média
            cat_cols = df.select_dtypes(include=["object"]).columns
            df[cat_cols] = df[cat_cols].fillna("Desconhecido")
            flash("Os valores nulos foram preenchidos automaticamente.")
        else:
            flash("Ação inválida. Nenhuma modificação foi feita.")
            return redirect(url_for("index"))

        # Verificar colunas obrigatórias
        required_columns = ["Make", "Model", "Clean Alternative Fuel Vehicle (CAFV) Eligibility"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
                flash(f"Erro: As colunas obrigatórias estão ausentes: {', '.join(missing_columns)}")
                return redirect(url_for("index"))

        # Salvar dataset tratado em arquivo temporário
        df.to_csv(file_path, index=False)

        # Treinar o modelo e gerar gráficos
        trained_model, label_encoder, accuracy, conf_matrix = train_model(df)

        # Gerar gráficos para o dataset tratado
        generate_all_visualizations(df)

        # Feedback ao usuário sobre o desempenho do modelo
        flash(f"Modelo treinado com acurácia de {accuracy:.2f}.")
        flash(f"Matriz de Confusão: {conf_matrix}")

        # Redirecionar para análise
        return redirect(url_for("analyze", file_path=file_path))
    except Exception as e:
        flash(f"Erro ao processar os valores nulos: {str(e)}")
        return redirect(url_for("index"))

@app.route("/download_report/<file_path>", methods=["GET"])
def download_report(file_path):
    global accuracy, conf_matrix
    try:
        print(f"Recebido em /download_report: {file_path}")
        # Geração do relatório
        pdf_path = generate_pdf_report(file_path, accuracy, conf_matrix)
        print(f"PDF gerado em: {pdf_path}")
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        print(f"Erro ao gerar o relatório: {str(e)}")
        flash(f"Erro ao gerar o relatório: {str(e)}")
        return redirect(url_for("analyze", file_path=file_path))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    global trained_model, label_encoder, loaded_dataset
    if trained_model is None or label_encoder is None:
        flash("O modelo ainda não foi treinado. Por favor, carregue o dataset primeiro.")
        return redirect(url_for("index"))

    if loaded_dataset is None:
        flash("O dataset ainda não foi carregado. Por favor, carregue-o antes de acessar esta funcionalidade.")
        return redirect(url_for("index"))

    unique_values = {
        "make": loaded_dataset["Make"].unique().tolist(),
        "model": loaded_dataset["Model"].unique().tolist(),
        "cafv": loaded_dataset["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].unique().tolist(),
        "model_year": sorted(loaded_dataset["Model Year"].dropna().unique().astype(int).tolist()),
        "electric_range": sorted(loaded_dataset["Electric Range"].dropna().unique().astype(int).tolist()),
        "base_msrp": sorted(loaded_dataset["Base MSRP"].dropna().unique().astype(float).tolist()),
    }

    if request.method == "POST":
        try:
            # Coletar valores do formulário
            model_year = int(float(request.form["model_year"]))
            electric_range = int(float(request.form["electric_range"]))
            base_msrp = float(request.form["base_msrp"])
            make = request.form["make"]
            model = request.form["model"]
            cafv = request.form["cafv"]

            input_data = pd.DataFrame({
                'Model Year': [model_year],
                'Electric Range': [electric_range],
                'Base MSRP': [base_msrp],
                'Make': [make],
                'Model': [model],
                'Clean Alternative Fuel Vehicle (CAFV) Eligibility': [cafv]
            })

            # Transformar variáveis categóricas
            input_data = pd.get_dummies(input_data)
            model_columns = trained_model.feature_importances_.shape[0]
            input_data = input_data.reindex(columns=range(model_columns), fill_value=0)

            # Fazer a predição
            prediction = trained_model.predict(input_data)
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            # Renderizar os resultados
            return render_template(
                "predict.html",
                prediction=predicted_class,
                input_data={
                    "Model Year": model_year,
                    "Electric Range": electric_range,
                    "Base MSRP": base_msrp,
                    "Make": make,
                    "Model": model,
                    "CAFV Eligibility": cafv,
                }  # Envia as entradas do usuário
            )
        except Exception as e:
            flash(f"Erro ao fazer a predição: {str(e)}")
            return redirect(url_for("predict"))

    return render_template("predict_form.html", unique_values=unique_values)

# Rota para análise de dados
@app.route("/analyze/<file_path>", methods=["GET", "POST"])
def analyze(file_path):
    global trained_model, label_encoder, loaded_dataset, selected_classifier
    global accuracy, conf_matrix  # Torna as variáveis globais

    try:
        # Carregar o dataset
        df = pd.read_csv(file_path)
        loaded_dataset = df

        # Obter lista de estados, anos e fabricantes se as colunas existirem
        states = df['State'].unique().tolist() if 'State' in df.columns else []
        unique_years = sorted(df['Model Year'].dropna().unique().astype(int).tolist()) if 'Model Year' in df.columns else []
        unique_makes = df['Make'].dropna().unique().tolist() if 'Make' in df.columns else []

        # Inicializar variáveis para filtros e mapas
        city_map_url = None
        state_selected = None
        year_selected = None
        make_selected = None

        # Obter hiperparâmetros configurados
        n_estimators = int(request.form.get("n_estimators", 100))  # Padrão: 100
        max_depth = request.form.get("max_depth")
        max_depth = int(max_depth) if max_depth else None  # Permite None se vazio
        random_state = 42  # Fixo para reprodutibilidade

        # Criar dicionário de hiperparâmetros
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        }

        # Gerar mapa interativo por estado
        if 'State' in df.columns:
            generate_map_by_region(df, region_col="State", value_col="Electric Vehicle Type")

            # Filtrar cidades pelo estado selecionado
            if request.method == "POST":
                state_selected = request.form.get("state")
                year_selected = request.form.get("year_filter")
                make_selected = request.form.get("make_filter")

                # Aplicar filtros no DataFrame
                if year_selected:
                    df = df[df['Model Year'] == int(year_selected)]
                if make_selected:
                    df = df[df['Make'] == make_selected]
                if state_selected and 'City' in df.columns and not df['City'].isnull().all():
                    generate_map_by_city(
                        df,
                        state_col="State",
                        city_col="City",
                        value_col="Electric Vehicle Type",
                        state_filter=state_selected
                    )
                    city_map_url = url_for('static', filename=f'map_by_city_{state_selected}.html')
                else:
                    flash("Não foi possível gerar o mapa por cidade. Verifique os dados.")

        # Verificar valores nulos
        if df.isnull().values.any():
            total_missing = df.isnull().sum().sum()
            flash(f"O dataset contém {total_missing} valores nulos.")
            return render_template("handle_nulls.html", file_path=file_path)

        # Treinar o modelo com os hiperparâmetros configurados
        trained_model, label_encoder, accuracy, conf_matrix = train_model(df, hyperparameters=hyperparameters)

        # Gerar gráficos com os dados filtrados
        generate_all_visualizations(df)

        # Renderizar resultados
        head_html = df.head().to_html()

        # Converter a matriz de confusão em tabela enumerada
        enumerated_conf_matrix = list(enumerate(conf_matrix.tolist()))

        return render_template(
            "results.html",
            table=head_html,
            classifier_name=selected_classifier,
            supported_classifiers=SUPPORTED_CLASSIFIERS,
            bar_chart_url="static/top_makes.png",
            pie_chart_url="static/vehicle_types.png",
            range_histogram_url="static/range_histogram.png",
            model_year_bar_chart_url="static/model_year_bar_chart.png",
            avg_range_by_make_url="static/avg_range_by_make.png",
            interactive_bar_chart_url="static/interactive_bar_chart.html",
            interactive_pie_chart_url="static/interactive_pie_chart.html",
            interactive_map_url="static/map_by_region.html",
            city_map_url=city_map_url,
            states=states,
            unique_years=unique_years,
            unique_makes=unique_makes,
            state_selected=state_selected,
            year_selected=year_selected,
            make_selected=make_selected,
            n_estimators = n_estimators,
            max_depth = max_depth,
            accuracy = accuracy,
            conf_matrix=enumerated_conf_matrix,  # Matriz de confusão enumerada
            file_path=file_path  # Adicione esta linha para garantir que o caminho do arquivo seja passado
        )
    except Exception as e:
        flash(f"Erro ao processar o arquivo: {str(e)}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)