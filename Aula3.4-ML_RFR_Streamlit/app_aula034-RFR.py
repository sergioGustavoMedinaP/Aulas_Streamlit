import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Configuração da Página
st.set_page_config(page_title="ML Production Lab", layout="wide", page_icon="🤖")

st.title("💰 Dashboard de Produção: Modelos de Regressão")
st.markdown("Compare algoritmos de Machine Learning e realize predições em tempo real.")

# 2. Sidebar: Gestão de Dados
st.sidebar.header("📁 Configuração de Dados")
fonte_dados = st.sidebar.radio("Selecione o Dataset:", ["Usar roi.csv local", "Fazer upload de novo arquivo"])

df = None

if fonte_dados == "Usar roi.csv local":
    try:
        df = pd.read_csv("Dados2026/roi.csv")
        st.sidebar.success("Arquivo roi.csv carregado!")
    except:
        st.sidebar.error("Arquivo roi.csv não encontrado na pasta raiz.")
else:
    arquivo_upload = st.sidebar.file_uploader("Selecione o arquivo CSV", type="csv")
    if arquivo_upload:
        df = pd.read_csv(arquivo_upload)

# 3. Processamento Principal
if df is not None:
    # Padronização de colunas para evitar erros de digitação
    df.columns = [c.strip().title() for c in df.columns]
    
    # Criando as abas
    tab_analise, tab_calculadora = st.tabs(["📊 Análise de Modelos", "🔮 Predição Individual"])

    with tab_analise:
        st.subheader("📊 Visualização e Performance")
        
        # Seleção de colunas dinâmicas
        col_x = st.multiselect("Variáveis Independentes (X):", df.columns.tolist(), default=df.columns[:-1].tolist())
        col_y = st.selectbox("Variável Alvo (Y):", df.columns.tolist(), index=len(df.columns)-1)

        algos_selecionados = st.multiselect(
            "Escolha os algoritmos para comparar:",
            ["Dummy (DR)", "Regressão Simples (RLS)", "Regressão Múltipla (RLM)", "Random Forest (RFR)"],
            default=["Regressão Múltipla (RLM)", "Random Forest (RFR)"]
        )

        if st.button("🚀 Treinar e Comparar"):
            X = df[col_x]
            y = df[col_y]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            resultados = []
            resumo_preds = pd.DataFrame({'Real': y_test.values})

            # Treinamento dos modelos selecionados
            if "Dummy (DR)" in algos_selecionados:
                m = DummyRegressor(strategy='mean').fit(X_train, y_train)
                p = m.predict(X_test)
                resultados.append({"Algoritmo": "Dummy", "R2": r2_score(y_test, p), "MAE": mean_absolute_error(y_test, p)})
                resumo_preds["Dummy"] = p

            if "Regressão Simples (RLS)" in algos_selecionados:
                m = LinearRegression().fit(X_train[[col_x[0]]], y_train)
                p = m.predict(X_test[[col_x[0]]])
                resultados.append({"Algoritmo": "RLS", "R2": r2_score(y_test, p), "MAE": mean_absolute_error(y_test, p)})
                resumo_preds["RLS"] = p

            if "Regressão Múltipla (RLM)" in algos_selecionados:
                m = LinearRegression().fit(X_train, y_train)
                p = m.predict(X_test)
                resultados.append({"Algoritmo": "RLM", "R2": r2_score(y_test, p), "MAE": mean_absolute_error(y_test, p)})
                resumo_preds["RLM"] = p

            if "Random Forest (RFR)" in algos_selecionados:
                m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
                p = m.predict(X_test)
                resultados.append({"Algoritmo": "RFR", "R2": r2_score(y_test, p), "MAE": mean_absolute_error(y_test, p)})
                resumo_preds["RFR"] = p

            # Exibição de Métricas
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("### 🏆 Performance")
                st.dataframe(pd.DataFrame(resultados).sort_values(by="R2", ascending=False))
            with c2:
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=resumo_preds['Real'], y=resumo_preds['Real'], name="Ideal", line=dict(color='black', dash='dash')))
                for col in resumo_preds.columns[1:]:
                    fig_res.add_trace(go.Scatter(x=resumo_preds['Real'], y=resumo_preds[col], mode='markers', name=col, opacity=0.6))
                st.plotly_chart(fig_res, use_container_width=True)

    with tab_calculadora:
        st.subheader("🔮 Estimativa de Retorno em Tempo Real")
        st.info("Insira os valores abaixo para calcular a previsão baseada no modelo Random Forest.")
        
        # Treinamos o modelo com todos os dados para maior precisão na calculadora
        X_prod = df[col_x if 'col_x' in locals() else df.columns[:-1]]
        y_prod = df[col_y if 'col_y' in locals() else df.columns[-1]]
        modelo_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_prod, y_prod)

        # Formulário para entrada via teclado
        with st.form("form_calculo"):
            st.write("### Dados de Entrada")
            # Gerando campos de input dinamicamente baseados nas colunas X
            entradas_usuario = []
            col_form = st.columns(len(X_prod.columns))
            
            for i, col_name in enumerate(X_prod.columns):
                val = col_form[i].number_input(f"Valor de {col_name}", value=float(df[col_name].mean()))
                entradas_usuario.append(val)
            
            btn_prever = st.form_submit_button("💎 Estimar Retorno")

            if btn_prever:
                # Criando o DataFrame de entrada para o Predict
                input_df = pd.DataFrame([entradas_usuario], columns=X_prod.columns)
                predicao = modelo_final.predict(input_df)[0]
                
                st.divider()
                st.metric(label="Retorno Estimado", value=f"R$ {predicao:,.2f}")
                st.success(f"Com base no modelo RFR, o retorno esperado é de R$ {predicao:,.2f}")

else:
    st.warning("Aguardando carregamento do arquivo para iniciar as análises.")