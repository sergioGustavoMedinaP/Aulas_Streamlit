import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Configuração da página
st.set_page_config(page_title="Analytics Pro - Regressão", layout="wide")

st.title("📊 Dashboard de Performance e Alinhamento (45°)")

# --- SIDEBAR ---
st.sidebar.header("Painel de Configuração")
arquivo_postado = st.sidebar.file_uploader("Upload do Dataset (CSV)", type="csv")
algoritmos_escolhidos = st.sidebar.multiselect(
    "Selecione os Algoritmos para Análise",
    ["Random Forest", "Regressão Linear", "SVR (Linear)", "SVR (RBF)"],
    default=["Random Forest", "Regressão Linear"]
)

if arquivo_postado is not None:
    # 1. Preparação dos Dados
    df = pd.read_csv(arquivo_postado).dropna()
    if 'CustomerID' in df.columns: 
        df = df.drop(columns=['CustomerID'])
    
    col_cat = ['sexo', 'assinatura', 'duracao_contrato']
    df_ready = pd.get_dummies(df, columns=col_cat, drop_first=True, dtype=int)
    
    X = df_ready.drop(columns=['total_gasto'])
    y = df_ready['total_gasto']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tabela_resultados = []

    # Criando as Abas
    tab_metricas, tab_graficos, tab_predicao = st.tabs(["📋 Métricas", "📈 Gráficos 45°", "🔮 Simulador"])

    with tab_metricas:
        st.subheader("Performance Comparativa")
        for algo in algoritmos_escolhidos:
            # Seleção do Modelo
            if algo == "Random Forest": model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif algo == "Regressão Linear": model = LinearRegression()
            elif algo == "SVR (Linear)": model = SVR(kernel='linear')
            else: model = SVR(kernel='rbf')

            # Treino e Predição
            inicio = time.time()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            fim = time.time()
            tempo = (fim - inicio) * 1000

            # Cálculos
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            tabela_resultados.append({
                "Algoritmo": algo,
                "R² Score": round(r2, 4),
                "MAE (R$)": round(mae, 2),
                "RMSE (R$)": round(rmse, 2),
                "MAPE (%)": f"{mape:.2f}%",
                "Tempo (ms)": round(tempo, 2),
                "y_pred": y_pred # Guardando para os gráficos
            })

            # Exibição de Cartões
            st.write(f"**{algo}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score", round(r2, 4))
            c2.metric("MAE", f"R$ {mae:.2f}")
            c3.metric("RMSE", f"R$ {rmse:.2f}")
            c4.metric("MAPE (%)", f"{mape:.2f}%")
            st.divider()
        
        st.subheader("Tabela de Resumo Técnico")
        df_final = pd.DataFrame(tabela_resultados).drop(columns=['y_pred'])
        st.dataframe(df_final, use_container_width=True)

    with tab_graficos:
        st.subheader("Análise de Alinhamento (Real vs Predito)")
        st.write("A linha tracejada representa a **reta de 45 graus**. Quanto mais próximos os pontos estiverem dela, melhor.")
        
        cols_graf = st.columns(2)
        for idx, res in enumerate(tabela_resultados):
            with cols_graf[idx % 2]:
                y_p = res["y_pred"]
                
                # Criando o gráfico com Plotly Objects para ter controle total da linha de 45°
                fig = go.Figure()
                
                # Pontos de dispersão
                fig.add_trace(go.Scatter(x=y_test, y=y_p, mode='markers', name='Predições',
                                         marker=dict(color='blue', opacity=0.5)))
                
                # Reta de 45 graus (Linha Ideal)
                mn = min(y_test.min(), y_p.min())
                mx = max(y_test.max(), y_p.max())
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', 
                                         name='Linha 45° (Ideal)', line=dict(color='red', dash='dash')))
                
                fig.update_layout(title=f"Alinhamento: {res['Algoritmo']}",
                                  xaxis_title="Valor Real", yaxis_title="Valor Predito",
                                  height=400, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)

    with tab_predicao:
        st.subheader("Simulador de Novos Clientes")
        # Mantém a lógica de predição anterior aqui...

else:
    st.info("Aguardando upload do arquivo CSV para processar as métricas e os gráficos.")