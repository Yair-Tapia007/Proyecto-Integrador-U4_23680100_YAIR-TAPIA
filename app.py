
import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="AgroTech Analytics | Dashboard U4",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    /* Fondo sutil y tipografía */
    .main {
        background-color: #f8faf9;
    }
    
    /* Banner Principal con Degradado */
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 20px rgba(27, 67, 50, 0.18);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 10px;
        backdrop-filter: blur(4px);
    }

    /* Estilizado de Tarjetas KPI nativas */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2d6a4f;
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }

    /* Títulos de sección */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1b4332;
        margin: 1rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #d8f3dc;
    }

    /* Caja de veredicto */
    .verdict-box {
        background: linear-gradient(90deg, #e9f5ed 0%, #f4fbf7 100%);
        border-left: 6px solid #198754;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        color: #0f5132;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- ENCABEZADO HERO ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌿 Optimización Inferencial del Rendimiento Agrícola y Control de Riego</div>
    <div class="hero-subtitle">Plataforma de Telemetría Agroindustrial en Invernadero Inteligente</div>
    <div class="hero-badge">⚡ Motor DuckDB (Semilla 0.42) &nbsp;|&nbsp; 📈 Regresión Múltiple OLS &nbsp;|&nbsp; 🧪 Inferencia t-Student</div>
</div>
""", unsafe_allow_html=True)

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

# --- BARRA LATERAL: CONTROLES Y SIMULADOR ---
with st.sidebar:
    st.markdown("## 🎛️ Panel de Control")
    st.caption("Ajusta los parámetros en tiempo real para explorar las parcelas.")
    
    tratamientos_sel = st.multiselect(
        "🧪 Tipo de Tratamiento:",
        options=['Tratamiento_A', 'Tratamiento_B'],
        default=['Tratamiento_A', 'Tratamiento_B']
    )

    rango_riego = st.slider(
        "💧 Volumen de Riego (m³/ha):",
        min_value=float(df_global['volumen_riego'].min()),
        max_value=float(df_global['volumen_riego'].max()),
        value=(float(df_global['volumen_riego'].min()), float(df_global['volumen_riego'].max()))
    )

    rango_fert = st.slider(
        "🌾 Fertilizante Aplicado (kg/ha):",
        min_value=float(df_global['fertilizante_kg'].min()),
        max_value=float(df_global['fertilizante_kg'].max()),
        value=(float(df_global['fertilizante_kg'].min()), float(df_global['fertilizante_kg'].max()))
    )

    # Filtrar dataframe en tiempo real
    df_filtrado = df_global[
        (df_global['tipo_tratamiento'].isin(tratamientos_sel)) &
        (df_global['volumen_riego'].between(rango_riego[0], rango_riego[1])) &
        (df_global['fertilizante_kg'].between(rango_fert[0], rango_fert[1]))
    ]

    st.markdown("---")
    st.markdown("### 🔮 Simulador Predictivo OLS")
    st.caption("Estima la cosecha esperada según nuestro modelo ajustado:")
    sim_riego = st.number_input("Riego a aplicar (m³/ha)", 100.0, 300.0, 220.0, step=10.0)
    sim_fert = st.number_input("Fertilizante a aplicar (kg/ha)", 50.0, 150.0, 110.0, step=5.0)
    
    pred_sim = modelo_ols.params['const'] + modelo_ols.params['volumen_riego']*sim_riego + modelo_ols.params['fertilizante_kg']*sim_fert
    st.success(f"**Rendimiento Estimado:**\n### {pred_sim:.2f} ton/ha")

    st.markdown("---")
    st.caption("🎓 **Ingeniería en Sistemas Computacionales** | IT Cuautla")
    st.caption("👤 Yair Gustavo Tapia Vazquez (23680100)")

# --- TARJETAS KPI DE RESUMEN (Fase 5.1) ---
st.markdown('<div class="section-header">📊 Indicadores Clave de Desempeño (KPIs)</div>', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

rendimiento_prom = df_filtrado['rendimiento_ton'].mean() if not df_filtrado.empty else 0.0
dif_global = rendimiento_prom - df_global['rendimiento_ton'].mean() if not df_filtrado.empty else 0.0

kpi1.metric(
    label="🌱 Rendimiento Promedio",
    value=f"{rendimiento_prom:.2f} ton/ha",
    delta=f"{dif_global:+.2f} ton/ha vs Global ({len(df_filtrado)} parcelas)"
)

kpi2.metric(
    label="📈 R² Ajustado (Modelo OLS)",
    value=f"{r2_ajustado * 100:.2f}%",
    delta=f"RMSE: {rmse_val:.3f} ton/ha"
)

kpi3.metric(
    label="🧪 P-Valor (Prueba t-Student)",
    value=f"{p_valor:.6f}",
    delta="Significativo (p < 0.05) ✔" if p_valor < 0.05 else "No Significativo"
)

kpi4.metric(
    label="🚀 Bono Tratamiento B (IC 95%)",
    value=f"+{grupo_B.mean() - grupo_A.mean():.2f} ton/ha",
    delta=f"IC: [{ic_inf:.2f}, {ic_sup:.2f}] ton/ha"
)

st.write("")

# --- CUADRÍCULA 2x2 DE VISUALIZACIONES CON BORDE (Fase 4 y Fase 5.2) ---
st.markdown('<div class="section-header">📉 Diagnóstico Visual e Inferencial (Cuadrícula 2×2)</div>', unsafe_allow_html=True)

if df_filtrado.empty:
    st.warning("⚠️ No hay parcelas que coincidan con los filtros seleccionados. Ajusta los controles en el panel izquierdo.")
else:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    # Fila 1 de la cuadrícula 2x2
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("##### 🔥 1. Mapa de Calor de Correlación de Pearson")
            st.caption("Asociación lineal entre insumos agronómicos y toneladas cosechadas.")
            fig1, ax1 = plt.subplots(figsize=(6, 4.2))
            corr_matrix = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].rename(
                columns={'volumen_riego': 'Riego (m³)', 'fertilizante_kg': 'Fertilizante (kg)', 'rendimiento_ton': 'Rendimiento (t)'}
            ).corr(method='pearson')
            sns.heatmap(
                corr_matrix, annot=True, cmap='YlGnBu', fmt=".3f", vmin=-1, vmax=1, ax=ax1,
                linewidths=1.5, annot_kws={"size": 11, "weight": "bold"}, cbar_kws={'shrink': 0.8}
            )
            plt.tight_layout()
            st.pyplot(fig1)
        
    with col2:
        with st.container(border=True):
            st.markdown("##### 📈 2. Dispersión con Ajuste Lineal OLS (Y vs X)")
            st.caption("Tendencia predictiva por Mínimos Cuadrados Ordinarios con banda de confianza al 95%.")
            fig2, ax2 = plt.subplots(figsize=(6, 4.2))
            sns.regplot(
                data=df_filtrado, x='volumen_riego', y='rendimiento_ton', ax=ax2,
                scatter_kws={'alpha': 0.7, 'color': '#2d6a4f', 's': 45, 'edgecolor': 'white'},
                line_kws={'color': '#d90429', 'linewidth': 2.2, 'label': f'Recta OLS (R²={r2_ajustado:.2f})'}
            )
            ax2.set_xlabel("Volumen de Riego Aplicado (m³/ha)", fontweight='bold')
            ax2.set_ylabel("Rendimiento Logrado (ton/ha)", fontweight='bold')
            ax2.legend(frameon=True, facecolor='white')
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig2)

    # Fila 2 de la cuadrícula 2x2
    col3, col4 = st.columns(2)
    
    with col3:
        with st.container(border=True):
            st.markdown("##### 📦 3. Diagrama de Cajas Comparativo por Tratamiento")
            st.caption("Soporte visual de la prueba t-Student (el punto blanco indica la media de cada grupo).")
            fig3, ax3 = plt.subplots(figsize=(6, 4.2))
            sns.boxplot(
                data=df_filtrado, x='tipo_tratamiento', y='rendimiento_ton',
                hue='tipo_tratamiento', palette=['#74c69d', '#1b4332'], legend=False, ax=ax3, width=0.45,
                boxprops=dict(alpha=0.88),
                showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"#d90429", "markersize":"9", "markeredgewidth": 2}
            )
            sns.stripplot(
                data=df_filtrado, x='tipo_tratamiento', y='rendimiento_ton',
                color='black', alpha=0.25, jitter=0.15, size=5, ax=ax3
            )
            ax3.set_xlabel("Grupo de Tratamiento Experimental", fontweight='bold')
            ax3.set_ylabel("Rendimiento (ton/ha)", fontweight='bold')
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig3)
        
    with col4:
        with st.container(border=True):
            st.markdown("##### 🔔 4. Histograma y Densidad de Residuos (Diagnóstico)")
            st.caption("Validación del supuesto de normalidad del error alrededor de media cero.")
            fig4, ax4 = plt.subplots(figsize=(6, 4.2))
            sns.histplot(
                df_filtrado['residuos'], kde=True, color='#40916c', bins=14, ax=ax4,
                edgecolor='white', alpha=0.65, line_kws={'linewidth': 2.2, 'color': '#1b4332'}
            )
            ax4.axvline(0, color='#d90429', linestyle='--', linewidth=2, label='Media Cero (Error = 0)')
            ax4.set_xlabel("Residuos del Modelo (Y_real - Y_predicho)", fontweight='bold')
            ax4.set_ylabel("Frecuencia de Parcelas", fontweight='bold')
            ax4.legend(frameon=True, facecolor='white')
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig4)

st.write("")

# --- SECCIÓN INFERIOR CON PESTAÑAS INTERACTIVAS ---
st.markdown('<div class="section-header">📑 Documentación Técnica, Ecuación OLS y Base de Datos</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🏆 Veredicto de Ingeniería (t-Student)", 
    "📐 Resumen Matemático OLS", 
    "🗄️ Explorador de Telemetría SQL"
])

with tab1:
    st.markdown(f"""
    <div class="verdict-box">
        <h4 style="margin-top:0; color:#0f5132;">✅ Decisión Técnica: Autorizar Escalamiento del Bio-estimulante (Tratamiento B)</h4>
        <p style="margin-bottom:0.5rem;">
            Al contrastar las hipótesis <b>H₀: μ_B - μ_A ≤ 0</b> contra <b>H₁: μ_B - μ_A > 0</b> mediante la prueba <i>t-Student</i> para muestras independientes, 
            se obtuvo un estadístico <b>t = {t_stat:.4f}</b> y un <b>p-valor = {p_valor:.6f} &lt; α (0.05)</b>.
        </p>
        <p style="margin-bottom:0;">
            Por lo tanto, <b>se rechaza la Hipótesis Nula (H₀)</b>. Con un 95% de confianza, el <b>Tratamiento B</b> genera un incremento adicional de entre 
            <b>{ic_inf:.2f} y {ic_sup:.2f} ton/ha</b> (media observada de <b>+{grupo_B.mean() - grupo_A.mean():.2f} ton/ha</b> sobre el Tratamiento A), 
            justificando técnica y financieramente su adopción en todo el invernadero.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    col_eq, col_tab = st.columns([1, 1])
    with col_eq:
        st.markdown("#### Ecuación Predictiva Ajustada")
        b0 = modelo_ols.params['const']
        b1 = modelo_ols.params['volumen_riego']
        b2 = modelo_ols.params['fertilizante_kg']
        st.latex(rf"\widehat{{\text{{Rendimiento}}}} = {b0:.3f} + {b1:.4f}(\text{{Riego}}) + {b2:.4f}(\text{{Fertilizante}})")
        st.info(
            f"**Interpretación Agronómica:**\n"
            f"- El modelo explica el **{r2_ajustado*100:.2f}%** de la variabilidad de la cosecha.\n"
            f"- Cada $100\\text{{ m}}^3/\\text{{ha}}$ adicionales de riego incrementan el rendimiento en **{b1*100:.2f} ton/ha** ($p < 0.001$).\n"
            f"- Cada $50\\text{{ kg}}/\\text{{ha}}$ adicionales de fertilizante aportan **{b2*50:.2f} ton/ha** ($p < 0.001$)."
        )
    with col_tab:
        st.markdown("#### Coeficientes del Modelo (`statsmodels`)")
        resumen_df = pd.DataFrame({
            'Coeficiente (β)': modelo_ols.params.round(4),
            'Error Estándar': modelo_ols.bse.round(4),
            'Estadístico t': modelo_ols.tvalues.round(3),
            'P-Valor': modelo_ols.pvalues.apply(lambda x: f"{x:.4e}")
        })
        st.dataframe(resumen_df, use_container_width=True)

with tab3:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(f"Mostrando **{len(df_filtrado)} registros** extraídos en memoria con `DuckDB` (`setseed(0.42)`).")
    with col_btn:
        csv_data = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV Filtrado",
            data=csv_data,
            file_name="telemetria_invernadero_u4.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.dataframe(
        df_filtrado.style.background_gradient(subset=['rendimiento_ton'], cmap='Greens'),
        use_container_width=True,
        height=320
    )
