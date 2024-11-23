## Nome do Projeto
Uma breve descrição do que o projeto faz.

## Funcionalidades
- Funcionalidade 1
- Funcionalidade 2
- Funcionalidade 3

## Como Usar
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git

## Configuração do Ambiente:

### Instalação Flask:
- pip install flask pandas scikit-learn matplotlib plotly

### Reinstalar dependências no futuro:
- pip install -r requirements.txt

### Executar o Servidor Flask (Rodar o projeto)
- python app.py

### Implementar Gráficos Interativos
- pip install plotly



## Validações

Validação no Upload do Dataset
Antes de processar o arquivo CSV, valide se ele contém as colunas esperadas e está no formato correto. Se algo estiver errado, exiba uma mensagem amigável ao usuário.
Validação no Upload:

Verifica se todas as colunas obrigatórias estão presentes no dataset.
Gera mensagens de erro amigáveis.


Validação nos Valores do Dataset
Valide os valores do dataset para garantir que estão no formato correto (sem valores nulos ou inválidos).
Verifica se há valores nulos.
Valida se as colunas numéricas realmente contêm números.


Validação no Formulário de Predição
Valide os valores inseridos no formulário antes de enviá-los ao modelo de predição.

Garante que os valores inseridos pelo usuário estejam dentro de intervalos aceitáveis.
Exibe mensagens de erro específicas para valores inválidos.
