"""
EC1 - Modelamiento Predictivo de Datos - Grupo 8
Caso adaptado: Variabilidad e incumplimiento en la entrega de pedidos
de e-commerce y su impacto en la satisfacción del cliente y los costos logísticos.
Dataset: Brazilian E-Commerce Public Dataset by Olist (consolidado a nivel de pedido).

Ejecutar con:  streamlit run app.py
Requiere que 'olist_consolidado_pedido.csv' esté en la misma carpeta.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score, silhouette_score
)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
sns.set_style("whitegrid")

st.set_page_config(
    page_title="Grupo 8 - Modelamiento Predictivo | Olist",
    layout="wide",
)

st.title("📦 Variabilidad e incumplimiento en la entrega de pedidos de e-commerce")
st.caption(
    "Caso adaptado — Curso Modelamiento Predictivo de Datos | "
    "Dataset: Brazilian E-Commerce Public Dataset by Olist"
)

@st.cache_data
def cargar_datos():
    columnas_fecha = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    if os.path.exists("olist_consolidado_pedido.csv"):
        df = pd.read_csv("olist_consolidado_pedido.csv", parse_dates=columnas_fecha)
    elif os.path.exists("olist_consolidado_pedido.xlsx"):
        df = pd.read_excel("olist_consolidado_pedido.xlsx", parse_dates=columnas_fecha)
    else:
        raise FileNotFoundError
    return df

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error(
        "⚠️ No se encontró 'olist_consolidado_pedido.csv' ni 'olist_consolidado_pedido.xlsx'. "
        "Coloca uno de los dos archivos en la misma carpeta que este script antes de ejecutarlo."
    )
    st.stop()

df_entregados = df[df["late_delivery"].notna()].copy()
df_entregados["late_delivery"] = df_entregados["late_delivery"].astype(int)

df_kpi4 = df_entregados[df_entregados["review_score"].notna()].copy()
df_kpi4["satisfaccion_bin"] = (df_kpi4["review_score"] >= 4).astype(int)

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ OTD — Redes Neuronales",
    "2️⃣ Variabilidad — Series de Tiempo",
    "3️⃣ Sobrecosto — Clustering",
    "4️⃣ Satisfacción — Árbol / Random Forest",
])

# ============================================================
# TAB 1 - DIMENSIÓN 1: CUMPLIMIENTO DE ENTREGAS (OTD)
# REDES NEURONALES - PERCEPTRÓN MULTICAPA (MLP)
# ============================================================

with tab1:

    st.header("Dimensión 1: Cumplimiento de entregas (OTD)")

    st.markdown(
        "**Indicador:** OTD (On-Time Delivery)  \n"
        "**Técnica:** Redes neuronales - Perceptrón Multicapa (MLP)  \n"
        "**Proceso asociado:** Planificación y control de la entrega"
    )

    df_modelo = df[df["late_delivery"].notna()].copy()

    df_modelo["OTD"] = 1 - df_modelo["late_delivery"]

    variables = [
        "freight_value",
        "price",
        "payment_value_total",
        "payment_installments_max",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
        "n_items",
        "n_sellers",
        "n_products",
        "payment_type_main",
        "seller_state",
        "customer_state"
    ]

    X = df_modelo[variables].copy()
    y = df_modelo["OTD"].copy()

    st.subheader("Distribución del cumplimiento de entregas")

    otd_counts = y.value_counts().sort_index()

    st.bar_chart(
        pd.DataFrame({
            "Pedidos": otd_counts.values
        }, index=["Fuera de plazo", "Dentro de plazo"])
    )

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns

    preprocesador = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    X_train = preprocesador.fit_transform(X_train)
    X_test = preprocesador.transform(X_test)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential([
        Dense(
            64,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),
        Dropout(0.30),

        Dense(
            32,
            activation="relu"
        ),
        Dropout(0.30),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="AUC")
        ]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    y_prob = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("Resultados del modelo predictivo")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")
    col5.metric("AUC-ROC", f"{auc:.2%}")

    st.subheader("Matriz de confusión")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Fuera de plazo", "Dentro de plazo"],
        columns=["Predicho fuera", "Predicho dentro"]
    )

    st.dataframe(cm_df)

    st.subheader("Importancia de variables")

    import numpy as np

    base_auc = roc_auc_score(
        y_test,
        model.predict(X_test, verbose=0).ravel()
    )

    importancias = []

    for i in range(X_test.shape[1]):

        X_perm = X_test.copy()

        np.random.seed(42)
        X_perm[:, i] = np.random.permutation(
            X_perm[:, i]
        )

        auc_perm = roc_auc_score(
            y_test,
            model.predict(
                X_perm,
                verbose=0
            ).ravel()
        )

        importancias.append(
            base_auc - auc_perm
        )

    nombres = preprocesador.get_feature_names_out()

    importancia_df = pd.DataFrame({
        "Variable": nombres,
        "Importancia": importancias
    })

    importancia_df = (
        importancia_df
        .sort_values(
            "Importancia",
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        importancia_df.set_index("Variable")
    )

    st.subheader("Predicción de cumplimiento de entrega")

    st.write(
        "Ingrese las características del pedido para estimar "
        "la probabilidad de cumplimiento del plazo."
    )

    st.info(
        "El modelo genera una probabilidad estimada de que "
        "el pedido sea entregado dentro del plazo establecido."
    )

    st.markdown(
        "**Modelo:** Perceptrón Multicapa (MLP)  \n"
        "**Arquitectura:** 64 → 32 → 1 neuronas  \n"
        "**Activaciones:** ReLU → ReLU → Sigmoid  \n"
        "**Dropout:** 0.30  \n"
        "**Optimizador:** Adam"
    )

# ============================================================
# TAB 2 — DIMENSIÓN 2: Variabilidad del tiempo de entrega
# Técnica: Series de tiempo — Suavización Exponencial Simple
# ============================================================

with tab2:

    st.header("Dimensión 2: Variabilidad del tiempo de entrega")

    st.markdown(
        "**Indicador:** Variabilidad del tiempo de entrega &nbsp;|&nbsp; "
        "**Técnica:** Series de tiempo &nbsp;|&nbsp; "
        "**Modelo:** Suavización Exponencial Simple (SES)"
    )

    df2 = df[
        (df["order_status"] == "delivered") &
        (df["order_delivered_customer_date"].notna())
    ].copy()

    df2["dias_entrega_real"] = (
        df2["order_delivered_customer_date"] -
        df2["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    df2["dias_entrega_estimado"] = (
        df2["order_estimated_delivery_date"] -
        df2["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    df2["desviacion_entrega"] = (
        df2["dias_entrega_real"] -
        df2["dias_entrega_estimado"]
    )

    df2["mes"] = df2["order_purchase_timestamp"].dt.to_period("M")

    resumen_mensual_2 = (
        df2.groupby("mes")["desviacion_entrega"]
        .agg(variabilidad_desviacion="std")
        .reset_index()
    )

    resumen_mensual_2["mes"] = pd.to_datetime(
        resumen_mensual_2["mes"].astype(str)
    )

    resumen_mensual_2 = (
        resumen_mensual_2
        .sort_values("mes")
        .dropna()
    )

    st.subheader("Evolución mensual de la variabilidad")

    fig1, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(
        resumen_mensual_2["mes"],
        resumen_mensual_2["variabilidad_desviacion"],
        marker="o"
    )

    ax1.set_title(
        "Evolución mensual de la variabilidad del tiempo de entrega"
    )
    ax1.set_xlabel("Mes")
    ax1.set_ylabel("Variabilidad (días)")
    ax1.grid(True)

    st.pyplot(fig1)

    serie_2 = resumen_mensual_2[
        ["mes", "variabilidad_desviacion"]
    ].set_index("mes")

    train_2 = serie_2.iloc[:-5]
    test_2 = serie_2.iloc[-5:]

    train_val_2 = train_2.iloc[:-5]
    validacion_2 = train_2.iloc[-5:]

    resultados_alpha_2 = []

    for alpha in np.arange(0.05, 1.00, 0.05):

        modelo = SimpleExpSmoothing(
            train_val_2["variabilidad_desviacion"].reset_index(drop=True),
            initialization_method="estimated"
        ).fit(
            smoothing_level=alpha,
            optimized=False
        )

        pronostico_validacion = modelo.forecast(
            len(validacion_2)
        )

        mse_alpha = mean_squared_error(
            validacion_2["variabilidad_desviacion"],
            pronostico_validacion
        )

        resultados_alpha_2.append(
            [alpha, mse_alpha]
        )

    tabla_alpha_2 = pd.DataFrame(
        resultados_alpha_2,
        columns=["Alpha", "MSE"]
    )

    mejor_alpha = tabla_alpha_2.loc[
        tabla_alpha_2["MSE"].idxmin(),
        "Alpha"
    ]

    modelo_final_2 = SimpleExpSmoothing(
        train_2["variabilidad_desviacion"].reset_index(drop=True),
        initialization_method="estimated"
    ).fit(
        smoothing_level=mejor_alpha,
        optimized=False
    )

    pronostico_final_2 = modelo_final_2.forecast(
        len(test_2)
    )

    pronostico_final_2.index = test_2.index

    st.subheader("Modelo de Suavización Exponencial Simple")

    st.metric(
        "Parámetro de suavizamiento seleccionado (α)",
        f"{mejor_alpha:.2f}"
    )

    mae_2 = mean_absolute_error(
        test_2["variabilidad_desviacion"],
        pronostico_final_2
    )

    mse_2 = mean_squared_error(
        test_2["variabilidad_desviacion"],
        pronostico_final_2
    )

    rmse_2 = np.sqrt(mse_2)

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae_2:.4f}")
    c2.metric("MSE", f"{mse_2:.4f}")
    c3.metric("RMSE", f"{rmse_2:.4f}")

    st.subheader("Pronóstico de la variabilidad mensual")

    resultados_2 = pd.DataFrame({
        "Mes": test_2.index,
        "Valor real": test_2["variabilidad_desviacion"].values,
        "Pronóstico SES": pronostico_final_2.values
    })

    st.dataframe(
        resultados_2,
        use_container_width=True
    )

    st.subheader("Valores reales vs. pronóstico")

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(
        test_2.index,
        test_2["variabilidad_desviacion"],
        marker="o",
        label="Valor real"
    )

    ax2.plot(
        pronostico_final_2.index,
        pronostico_final_2,
        marker="o",
        label="Pronóstico SES"
    )

    ax2.set_title(
        "Valores reales vs. pronóstico final de la variabilidad mensual"
    )
    ax2.set_xlabel("Mes")
    ax2.set_ylabel("Variabilidad (días)")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

    st.info(
        f"El modelo SES seleccionó un parámetro α = {mejor_alpha:.2f}. "
        f"En el conjunto de prueba se obtuvo un MAE de {mae_2:.4f} días "
        f"y un RMSE de {rmse_2:.4f} días."
    )

with tab3:
    st.header("Dimensión 3: Índice de sobrecosto por demoras")
    st.markdown(
        "**Indicador:** Índice de sobrecosto por demoras &nbsp;|&nbsp; "
        "**Técnica:** Clustering (K-medias) &nbsp;|&nbsp; "
        "**Proceso asociado:** Gestión de costos logísticos"
    )

    df3 = df[
        (df["order_status"] == "delivered")
        & (df["order_delivered_customer_date"].notna())
    ].copy()

    st.subheader("1. Índice de sobrecosto por demoras")

    flete_a_tiempo = df3.loc[df3["late_delivery"] == 0, "freight_value"].mean()
    flete_tardio = df3.loc[df3["late_delivery"] == 1, "freight_value"].mean()
    sobrecosto_global = (flete_tardio - flete_a_tiempo) / flete_a_tiempo * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Flete promedio – a tiempo", f"R$ {flete_a_tiempo:.2f}")
    c2.metric("Flete promedio – tardío", f"R$ {flete_tardio:.2f}")
    c3.metric("Sobrecosto por demora", f"{sobrecosto_global:.2f}%")

    benchmark = df3[df3["late_delivery"] == 0].groupby("seller_state")["freight_value"].mean()
    df3["freight_benchmark"] = df3["seller_state"].map(benchmark)
    df3["indice_sobrecosto"] = (
        (df3["freight_value"] - df3["freight_benchmark"]) / df3["freight_benchmark"] * 100
    )

    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df3["indice_sobrecosto"].clip(-100, 200), bins=50, edgecolor="black")
    axes[0].set_title("Distribución del índice de sobrecosto")
    axes[0].set_xlabel("Índice de sobrecosto (%)")
    axes[0].set_ylabel("Cantidad de pedidos")
    axes[0].grid(axis="y", alpha=0.3)

    sns.boxplot(
        x=df3["late_delivery"].map({0: "A tiempo", 1: "Tardío"}),
        y=df3["freight_value"].clip(upper=150),
        ax=axes[1],
    )
    axes[1].set_title("Costo de flete: a tiempo vs. tardío")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Costo de flete (R$)")

    plt.tight_layout()
    st.pyplot(fig1)

    st.subheader("2. Variables agregadas por mercado (seller_state)")

    agg = (
        df3.groupby("seller_state")
        .agg(
            n_pedidos=("order_id", "count"),
            pct_tardios=("late_delivery", "mean"),
            freight_promedio=("freight_value", "mean"),
            price_promedio=("price", "mean"),
            indice_sobrecosto_promedio=("indice_sobrecosto", "mean"),
        )
        .reset_index()
    )
    agg["pct_tardios"] = agg["pct_tardios"] * 100
    agg = agg[agg["n_pedidos"] >= 10].reset_index(drop=True)

    st.dataframe(
        agg.sort_values("indice_sobrecosto_promedio", ascending=False).style.format(
            {
                "pct_tardios": "{:.2f}%",
                "freight_promedio": "R$ {:.2f}",
                "price_promedio": "R$ {:.2f}",
                "indice_sobrecosto_promedio": "{:.2f}%",
            }
        )
    )
    st.caption(
        f"Se analizan {agg.shape[0]} mercados (seller_state) con al menos 10 pedidos, "
        f"a partir de {len(df3):,} pedidos entregados."
    )

    st.subheader("3. Segmentación con K-medias")

    features = ["pct_tardios", "freight_promedio", "indice_sobrecosto_promedio", "price_promedio"]
    X = agg[features]
    X_scaled = StandardScaler().fit_transform(X)

    resultados_k = []
    for k in range(2, 6):
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        resultados_k.append((k, silhouette_score(X_scaled, km_k.labels_)))
    mejor_k, mejor_sil = max(resultados_k, key=lambda t: t[1])

    kmeans = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    st.markdown(
        f"Se evaluó *k* entre 2 y 5 mediante el coeficiente de silueta, seleccionando "
        f"**k = {mejor_k}** por presentar el mayor valor (silueta = {mejor_sil:.3f})."
    )

    fig_k, ax_k = plt.subplots(figsize=(5, 3.5))
    ks, sils = zip(*resultados_k)
    ax_k.plot(ks, sils, marker="o")
    ax_k.axvline(mejor_k, color="red", linestyle="--", alpha=0.6)
    ax_k.set_xlabel("Número de clusters (k)")
    ax_k.set_ylabel("Coeficiente de silueta")
    ax_k.set_title("Selección de k")
    ax_k.grid(alpha=0.3)
    st.pyplot(fig_k)

    st.subheader("4. Clusters de mercados y tabla resumen")

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    scatter = ax2.scatter(
        agg["pct_tardios"],
        agg["indice_sobrecosto_promedio"],
        c=agg["cluster"],
        cmap="viridis",
        s=agg["n_pedidos"] / agg["n_pedidos"].max() * 500 + 60,
        edgecolor="black",
        alpha=0.85,
    )
    for _, row in agg.iterrows():
        ax2.annotate(
            row["seller_state"],
            (row["pct_tardios"], row["indice_sobrecosto_promedio"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax2.set_xlabel("% de pedidos tardíos")
    ax2.set_ylabel("Índice de sobrecosto promedio (%)")
    ax2.set_title("Clusters de mercados (seller_state) según riesgo y sobrecosto")
    ax2.grid(alpha=0.3)
    legend1 = ax2.legend(*scatter.legend_elements(), title="Cluster")
    ax2.add_artist(legend1)
    st.pyplot(fig2)

    resumen_clusters = (
        agg.groupby("cluster")
        .agg(
            n_mercados=("seller_state", "count"),
            n_pedidos_total=("n_pedidos", "sum"),
            pct_tardios_prom=("pct_tardios", "mean"),
            sobrecosto_prom=("indice_sobrecosto_promedio", "mean"),
            freight_prom=("freight_promedio", "mean"),
        )
        .reset_index()
    )
    st.dataframe(
        resumen_clusters.style.format(
            {
                "pct_tardios_prom": "{:.2f}%",
                "sobrecosto_prom": "{:.2f}%",
                "freight_prom": "R$ {:.2f}",
            }
        )
    )

    centros = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    centros.T.plot(kind="bar", ax=ax3)
    ax3.set_title("Perfil de los clusters (valores estandarizados)")
    ax3.set_ylabel("Valor estandarizado")
    ax3.legend(title="Cluster")
    ax3.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

    st.subheader("5. Interpretación")

    cluster_riesgo = resumen_clusters.loc[resumen_clusters["sobrecosto_prom"].idxmax(), "cluster"]
    cluster_modelo = resumen_clusters.loc[resumen_clusters["sobrecosto_prom"].idxmin(), "cluster"]
    estados_riesgo = agg.loc[agg["cluster"] == cluster_riesgo, "seller_state"].tolist()
    estados_modelo = agg.loc[agg["cluster"] == cluster_modelo, "seller_state"].tolist()
    sobrecosto_riesgo = resumen_clusters.loc[resumen_clusters["cluster"] == cluster_riesgo, "sobrecosto_prom"].values[0]
    sobrecosto_modelo = resumen_clusters.loc[resumen_clusters["cluster"] == cluster_modelo, "sobrecosto_prom"].values[0]

    st.info(
        f"**Cluster {cluster_riesgo} — prioridad de atención:** agrupa a {', '.join(estados_riesgo)}, "
        f"con el mayor índice de sobrecosto promedio ({sobrecosto_riesgo:.2f}%). Son los mercados donde "
        "las demoras generan el mayor impacto económico en la distribución y deberían priorizarse en los "
        "planes de mejora logística (renegociación de tarifas de flete, cambio de transportista, etc.).\n\n"
        f"**Cluster {cluster_modelo} — referencia a replicar:** agrupa a {', '.join(estados_modelo)}, "
        f"con el menor sobrecosto promedio ({sobrecosto_modelo:.2f}%) y, en general, menor porcentaje de "
        "pedidos tardíos. Sus prácticas logísticas pueden tomarse como modelo para los demás clusters."
    )


with tab4:
    st.header("Dimensión 4: Brecha de satisfacción por demora")
    st.markdown(
        "**Indicador:** Brecha de satisfacción por demora &nbsp;|&nbsp; "
        "**Técnica:** Árbol de clasificación / Random Forest &nbsp;|&nbsp; "
        "**Proceso asociado:** Atención postventa y control de calidad del servicio"
    )

    rev_atiempo = df_kpi4[df_kpi4["late_delivery"] == 0]["review_score"].mean()
    rev_tarde = df_kpi4[df_kpi4["late_delivery"] == 1]["review_score"].mean()
    brecha = (rev_atiempo - rev_tarde) / rev_atiempo * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Review promedio (a tiempo)", f"{rev_atiempo:.2f} ★")
    col2.metric("Review promedio (tardío)", f"{rev_tarde:.2f} ★")
    col3.metric("Brecha de satisfacción", f"-{brecha:.2f}%")

    st.subheader("Análisis descriptivo")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_kpi4["review_score"], bins=5, ax=ax, color="#4C72B0")
        ax.set_title("Distribución de review_score")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        cross = pd.crosstab(df_kpi4["late_delivery"], df_kpi4["review_score"] >= 4,
                             normalize="index") * 100
        cross.columns = ["Mala (<4)", "Buena (>=4)"]
        cross.index = ["A tiempo", "Tarde"]
        cross.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
        ax.set_title("% Satisfacción según puntualidad")
        ax.set_ylabel("% de pedidos")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.subheader("Modelo predictivo: Random Forest")

    @st.cache_resource
    def entrenar_modelo_satisfaccion(data):
        d = data.copy()
        for c in ["seller_state", "payment_type_main"]:
            d[c + "_enc"] = LabelEncoder().fit_transform(d[c].astype(str))

        features = [
            "late_delivery", "seller_state_enc", "freight_value",
            "price", "payment_value_total", "payment_type_main_enc",
        ]
        X = d[features].fillna(0)
        y = d["satisfaccion_bin"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            criterion="gini", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        return rf, features, X_test, y_test, y_pred, d

    with st.spinner("Entrenando Random Forest..."):
        rf, feat_rf, Xte4, yte4, ypred4, d4 = entrenar_modelo_satisfaccion(df_kpi4)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Matriz de confusión**")
        cm = confusion_matrix(yte4, ypred4)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Mala", "Buena"], yticklabels=["Mala", "Buena"], ax=ax)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        st.pyplot(fig)
    with c2:
        st.write("**Métricas del modelo**")
        st.text(classification_report(yte4, ypred4, target_names=["Mala", "Buena"]))
        st.metric("F1-score macro", f"{f1_score(yte4, ypred4, average='macro'):.2f}")

    st.write("**Importancia de variables**")
    importancias = pd.Series(rf.feature_importances_, index=feat_rf).sort_values()
    labels_es = {
        "late_delivery": "Retraso en entrega",
        "seller_state_enc": "Estado del vendedor",
        "freight_value": "Costo de flete",
        "price": "Precio del producto",
        "payment_value_total": "Monto total pagado",
        "payment_type_main_enc": "Tipo de pago",
    }
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([labels_es[i] for i in importancias.index], importancias.values, color="#4C72B0")
    ax.set_xlabel("Importancia (Random Forest)")
    st.pyplot(fig)

    st.write("**Prueba el modelo con tus propios valores:**")
    ci1, ci2, ci3 = st.columns(3)
    late_in = ci1.selectbox("¿El pedido llegó tarde?", ["No", "Sí"])
    freight_in4 = ci2.number_input("Costo de flete (R$)", 0.0, 2000.0, 20.0, key="freight4")
    price_in4 = ci3.number_input("Precio del producto (R$)", 0.0, 20000.0, 100.0, key="price4")

    ci4, ci5 = st.columns(2)
    pay_in4 = ci4.number_input("Monto total pagado (R$)", 0.0, 20000.0, 120.0, key="pay4")
    pay_type_in = ci5.selectbox("Tipo de pago", ["credit_card", "boleto", "voucher", "debit_card"])

    if st.button("Predecir satisfacción"):
        late_val = 1 if late_in == "Sí" else 0
        entrada = pd.DataFrame([{
            "late_delivery": late_val,
            "seller_state_enc": 0,
            "freight_value": freight_in4,
            "price": price_in4,
            "payment_value_total": pay_in4,
            "payment_type_main_enc": 0,
        }])[feat_rf]
        pred = rf.predict(entrada)[0]
        proba = rf.predict_proba(entrada)[0][1]
        if pred == 1:
            st.success(f"😊 Predicción: calificación BUENA (probabilidad: {proba*100:.1f}%)")
        else:
            st.error(f"😞 Predicción: calificación MALA (probabilidad de ser buena: {proba*100:.1f}%)")

st.markdown("---")
st.caption(
    "Grupo 8 — Modelamiento Predictivo de Datos | Universidad de Lima | "
    "Fuente: Brazilian E-Commerce Public Dataset by Olist (Kaggle)"
)
