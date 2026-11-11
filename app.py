import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Optimización Agrícola - Dashboard U4",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Control de Riego")
st.markdown("**Proyecto Integrador U4** | Telemetría SQL (DuckDB), Regresión OLS e Inferencia Estadística")

# --- FASE 1: INGESTA DETERMINISTA CON DUCKDB ---
@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    
    query = """
    CREATE TABLE telemetria AS
    WITH DatosSimulados AS (
        SELECT 
            range + 1 AS parcela_id,
            ROUND(100.0 + random() * (300.0 - 100.0), 2) AS volumen_riego,
            ROUND(50.0 + random() * (150.0 - 50.0), 2) AS fertilizante_kg,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            (random() * 1.6) - 0.8 AS ruido
        FROM range(100)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
            CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END + ruido, 
        2) AS rendimiento_ton
    FROM DatosSimulados;
    """
    con.execute(query)
    return con.execute("SELECT * FROM telemetria").df()

df_global = cargar_datos()

# --- MODELO OLS E INFERENCIA GLOBAL ---
X_global = sm.add_constant(df_global[['volumen_riego', 'fertilizante_kg']])
Y_global = df_global['rendimiento_ton']
modelo_ols = sm.OLS(Y_global, X_global).fit()

df_global['prediccion_ols'] = modelo_ols.predict(X_global)
df_global['residuos'] = modelo_ols.resid
r2_ajustado = modelo_ols.rsquared_adj
rmse_val = np.sqrt(mean_squared_error(Y_global, df_global['prediccion_ols']))

# Prueba t-Student
grupo_A = df_global[df_global['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_B = df_global[df_global['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']
t_stat, p_valor = stats.ttest_ind(grupo_B, grupo_A, alternative='greater')

cm = sm.stats.CompareMeans(sm.stats.DescrStatsW(grupo_B), sm.stats.DescrStatsW(grupo_A))
ic_inf, ic_sup = cm.tconfint_diff(alpha=0.05, usevar='pooled')

# --- BARRA LATERAL: CONTROLES INTERACTIVOS (Hito 4.2 / Fase 5.2) ---
st.sidebar.header("⚙️ Filtros de Telemetría")

tratamientos_sel = st.sidebar.multiselect(
    "Seleccionar Tratamiento:",
    options=['Tratamiento_A', 'Tratamiento_B'],
    default=['Tratamiento_A', 'Tratamiento_B']
)

rango_riego = st.sidebar.slider(
    "Volumen de Riego (m³/ha):",
    min_value=float(df_global['volumen_riego'].min()),
    max_value=float(df_global['volumen_riego'].max()),
    value=(float(df_global['volumen_riego'].min()), float(df_global['volumen_riego'].max()))
)

# Filtrar dataframe en tiempo real
df_filtrado = df_global[
    (df_global['tipo_tratamiento'].isin(tratamientos_sel)) &
    (df_global['volumen_riego'].between(rango_riego[0], rango_riego[1]))
]

st.sidebar.markdown("---")
st.sidebar.metric("Parcelas Monitoreadas", f"{len(df_filtrado)} / 100")
st.sidebar.metric("RMSE del Modelo", f"{rmse_val:.3f} ton/ha")
st.sidebar.info(f"**IC 95% (Dif. B - A):**\n[{ic_inf:.2f}, {ic_sup:.2f}] ton/ha")

# --- TARJETAS KPI DE RESUMEN (Fase 5.1) ---
st.subheader("📊 Métricas Principales del Invernadero")
kpi1, kpi2, kpi3 = st.columns(3)

rendimiento_prom = df_filtrado['rendimiento_ton'].mean() if not df_filtrado.empty else 0.0

kpi1.metric(
    label="1. Rendimiento Promedio (Filtro Actual)",
    value=f"{rendimiento_prom:.2f} ton/ha",
    delta=f"{rendimiento_prom - df_global['rendimiento_ton'].mean():.2f} ton/ha vs Global"
)

kpi2.metric(
    label="2. R² Ajustado del Modelo OLS",
    value=f"{r2_ajustado * 100:.2f}%",
    delta="Variabilidad explicada (Riego + Fert.)"
)

kpi3.metric(
    label="3. P-Valor (Prueba t-Student B > A)",
    value=f"{p_valor:.6f}",
    delta="Significativo (p < 0.05) - Escalar Trat. B" if p_valor < 0.05 else "No Significativo"
)

st.markdown("---")

# --- CUADRÍCULA 2x2 DE VISUALIZACIONES (Fase 4 y Fase 5.2) ---
if df_filtrado.empty:
    st.warning("Selecciona al menos un tipo de tratamiento en el panel lateral para visualizar los datos.")
else:
    sns.set_theme(style="whitegrid")
    
    # Fila 1 de la cuadrícula 2x2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Mapa de Calor de Correlación (Pearson)")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        corr_matrix = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr(method='pearson')
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax1)
        st.pyplot(fig1)
        
    with col2:
        st.markdown("#### 2. Dispersión con Ajuste Lineal OLS")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.regplot(
            data=df_filtrado, x='volumen_riego', y='rendimiento_ton', ax=ax2,
            scatter_kws={'alpha': 0.6, 'color': '#2b5c8f'},
            line_kws={'color': '#d9534f', 'linewidth': 2, 'label': 'Ajuste OLS (IC 95%)'}
        )
        ax2.set_xlabel("Volumen de Riego (m³/ha)")
        ax2.set_ylabel("Rendimiento (ton/ha)")
        ax2.legend()
        st.pyplot(fig2)

    # Fila 2 de la cuadrícula 2x2
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 3. Diagrama de Cajas por Tratamiento (Soporte t-Student)")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(
            data=df_filtrado, x='tipo_tratamiento', y='rendimiento_ton',
            hue='tipo_tratamiento', palette='Set2', legend=False, ax=ax3, width=0.5,
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
        )
        ax3.set_xlabel("Tipo de Tratamiento")
        ax3.set_ylabel("Rendimiento (ton/ha)")
        st.pyplot(fig3)
        
    with col4:
        st.markdown("#### 4. Histograma y Densidad de Residuos (Diagnóstico)")
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(df_filtrado['residuos'], kde=True, color='#6f42c1', bins=15, ax=ax4)
        ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Media Cero')
        ax4.set_xlabel("Residuos (Error = Y_real - Y_predicho)")
        ax4.set_ylabel("Frecuencia")
        ax4.legend()
        st.pyplot(fig4)

# Sección desplegable con detalles técnicos y tabla de datos
with st.expander("🔎 Ver Resumen Estadístico OLS, Veredicto Inferencial y Tabla de Telemetría"):
    st.markdown("### Veredicto de Ingeniería Agroindustrial")
    st.success(
        f"**Decisión Técnica:** Rechazar Hipótesis Nula ($p = {p_valor:.6f} < 0.05$). "
        f"El **Tratamiento B (Bio-estimulante)** incrementa el rendimiento promedio en **+{grupo_B.mean() - grupo_A.mean():.2f} ton/ha** "
        f"(IC 95%: [{ic_inf:.2f}, {ic_sup:.2f}] ton/ha). **Se autoriza la inversión para escalarlo a todo el invernadero.**"
    )
    st.markdown("### Tabla de Datos Filtrados")
    st.dataframe(df_filtrado, use_container_width=True)
