# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS PARA MINIMIZACIÓN DE MERMAS*")
print("="*70)

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
# Cargar el dataset de mermas (ajustar nombre del archivo según corresponda)
try:
    data = pd.read_csv('mermas_dataset.csv')  # Cambiar por el nombre real de tu archivo
    print(f"✓ Dataset cargado exitosamente: {data.shape[0]} filas x {data.shape[1]} columnas")
    
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'mermas_dataset.csv'")
    print("Por favor, asegúrate de que el archivo esté en el directorio correcto")
    exit()

# Filtrar registros con merma_monto > -100 FILTRASO
#data = data[data['merma_monto'] > -100]

#edicion de los signos

data['merma_monto_p'] = data['merma_monto_p'] * np.sign(data['merma_monto'])

# PASO 3: EXPLORACIÓN INICIAL DEL DATASET DE MERMAS
print("\n=== EXPLORACIÓN INICIAL DEL DATASET ===")
print("\nPrimeras 5 filas del dataset:")
print(data.head())

print("\nInformación general del dataset:")
print(data.info())

print("\nEstadísticas descriptivas de las variables objetivo:")
if 'merma_unidad' in data.columns:
    print(f"Merma en unidades - Estadísticas:")
    print(data['merma_unidad'].describe())
    
if 'merma_monto' in data.columns:
    print(f"\nMerma en monto - Estadísticas:")
    print(data['merma_monto'].describe())

# Verificar valores faltantes
print(f"\nValores faltantes por columna:")
missing_data = data.isnull().sum()
print(missing_data[missing_data > 0])

# PASO 4: PREPROCESAMIENTO DE FECHAS
print("\n=== PREPROCESAMIENTO DE FECHAS ===")

# Convertir fecha a datetime si existe
if 'fecha' in data.columns:
    try:
        # Intentar diferentes formatos de fecha
        data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
        
        # Crear variables temporales adicionales
        data['dia_semana'] = data['fecha'].dt.dayofweek
        data['trimestre'] = data['fecha'].dt.quarter
        data['dia_mes'] = data['fecha'].dt.day
        
        print("✓ Variables temporales creadas: dia_semana, trimestre, dia_mes")
        
    except Exception as e:
        print(f"⚠️ Advertencia al procesar fechas: {e}")

# PASO 5: SELECCIÓN DE VARIABLE OBJETIVO
print("\n=== SELECCIÓN DE VARIABLE OBJETIVO ===")
target_variable = 'merma_monto'  # Definimos directamente la variable objetivo

# PASO 6: SELECCIÓN DE CARACTERÍSTICAS PREDICTORAS
print("\n=== SELECCIÓN DE CARACTERÍSTICAS PREDICTORAS ===")

# Corregir nombre de columna si es necesario
if 'ubicación_motivo' in data.columns:
    data = data.rename(columns={'ubicación_motivo': 'ubicacion_motivo'})

# Crear variables temporales adicionales
if 'fecha' in data.columns:
    try:
        data['mes_num'] = data['fecha'].dt.month
        print("✓ Variables temporales adicionales creadas")
    except Exception as e:
        print(f"Error procesando fechas para 'mes_num': {e}")




# Variables categóricas
categorical_features = ['negocio', 'seccion', 'linea', 'categoria', 
                        'abastecimiento', 'comuna', 'region', 'tienda', 
                        'zonal', 'motivo', 'ubicacion_motivo', 'mes', 'semestre']

# Variables numéricas
numeric_features = ['año', 'dia_semana', 'trimestre', 'dia_mes', 'mes_num']

# Filtrar solo las columnas que existen en el dataset
categorical_features = [col for col in categorical_features if col in data.columns]
numeric_features = [col for col in numeric_features if col in data.columns]

print(f"Características categóricas: {categorical_features}")
print(f"Características numéricas: {numeric_features}")

# Preparar datos para modelado
X = data[categorical_features + numeric_features]
y = data[target_variable]

# Eliminar filas con valores faltantes
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]
data = data[mask] # Asegúrate de que 'data' también se filtre para mantener la correspondencia de índices

print(f"Registros finales para modelado: {len(X)}")

# PASO 7: DIVISIÓN DE DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# PASO 8: PREPROCESAMIENTO
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

print("✓ Preprocesador configurado:")
print("   - Variables numéricas: Estandarización (StandardScaler)")
print("   - Variables categóricas: One-Hot Encoding")

# PASO 9: IMPLEMENTACIÓN DE MODELOS
print("\n=== IMPLEMENTACIÓN DE MODELOS ===")

# Modelo 1: Regresión Lineal (del ejemplo original)
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modelo 2: Random Forest (del ejemplo original)
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Modelo 3: Gradient Boosting (nuevo modelo adicional)
# Seleccionado porque es muy efectivo para problemas de regresión complejos
pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

# Modelo 4: K-Nearest Neighbors (del ejemplo original, opcional)
pipeline_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

print("✓ Modelos configurados:")
print("   1. Regresión Lineal")
print("   2. Random Forest")
print("   3. Gradient Boosting (nuevo)")
print("   4. K-Nearest Neighbors")

# PASO 10: ENTRENAMIENTO DE MODELOS
print("\n=== ENTRENAMIENTO DE MODELOS ===")

models = {
    'Regresión Lineal': pipeline_lr,
    'Random Forest': pipeline_rf,
    'Gradient Boosting': pipeline_gb,
    'K-Nearest Neighbors': pipeline_knn
}

trained_models = {}
for name, model in models.items():
    print(f"Entrenando {name}...")
    try:
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} entrenado exitosamente")
    except Exception as e:
        print(f"❌ Error entrenando {name}: {e}")

print(f"\n✓ {len(trained_models)} modelos entrenados correctamente")

# PASO 11: EVALUACIÓN DE MODELOS
print("\n=== EVALUACIÓN DE MODELOS ===")

results = {}
predictions = {}

for name, model in trained_models.items():
    try:
        # Realizar predicciones
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        print(f"✓ {name} evaluado")
        
    except Exception as e:
        print(f"❌ Error evaluando {name}: {e}")

# PASO 12: PRESENTACIÓN DE RESULTADOS
print("\n=== RESULTADOS DE LA EVALUACIÓN ===")

# Crear DataFrame con resultados
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print("\nComparación de métricas entre modelos:")
print(results_df)

# Identificar el mejor modelo
best_model_name = results_df['R²'].idxmax()
best_r2 = results_df.loc[best_model_name, 'R²']
best_rmse = results_df.loc[best_model_name, 'RMSE']

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"   R² = {best_r2:.4f} (explica {best_r2*100:.1f}% de la variabilidad)")
print(f"   RMSE = {best_rmse:.2f} (error promedio de predicción)")

# PASO 13: ANÁLISIS ESPECÍFICO PARA MERMAS
print("\n=== ANÁLISIS ESPECÍFICO PARA GESTIÓN DE MERMAS ===")

print("🎯 INTERPRETACIÓN PARA MINIMIZACIÓN DE MERMAS:")
print(f"• El modelo {best_model_name} puede explicar {best_r2*100:.1f}% de la variación en {target_variable}")
print(f"• Error promedio de predicción: ±{best_rmse:.2f} unidades de {target_variable}")

# Análisis de distribución de errores
best_predictions = predictions[best_model_name]
errors = y_test - best_predictions
relative_errors = (errors / y_test) * 100

print(f"\n📊 ANÁLISIS DE ERRORES DEL MEJOR MODELO:")
print(f"• Error absoluto promedio: {np.mean(np.abs(errors)):.2f}")
print(f"• Error relativo promedio: {np.mean(np.abs(relative_errors)):.1f}%")
print(f"• Rango de errores: [{errors.min():.2f}, {errors.max():.2f}]")

# Análisis de impacto económico potencial
if 'merma_monto' in target_variable:
    total_merma_real = y_test.sum()
    total_merma_predicha = best_predictions.sum()
    diferencia_total = abs(total_merma_real - total_merma_predicha)
    
    print(f"\n💰 IMPACTO ECONÓMICO POTENCIAL:")
    print(f"• Merma total real en test: ${total_merma_real:,.2f}")
    print(f"• Merma total predicha: ${total_merma_predicha:,.2f}")
    print(f"• Diferencia absoluta: ${diferencia_total:,.2f}")

# PASO 14: GUARDAR RESULTADOS DETALLADOS
print("\n=== GENERACIÓN DE REPORTES ===")

# Crear DataFrame con predicciones detalladas
detailed_results = pd.DataFrame({
    'Valor_Real': y_test,
    'Mejor_Prediccion': best_predictions,
    'Error_Absoluto': np.abs(y_test - best_predictions),
    'Error_Relativo_Pct': np.abs((y_test - best_predictions) / y_test) * 100
})

# Añadir información contextual
# Para asegurar que los índices de `X_test` y `data` se alineen correctamente
# crea una copia de `X_test` con el índice original para poder usarlo para lookup en `data`
X_test_original_index = X_test.index 
original_data_for_lookup = data.loc[X_test_original_index]

# Agregar variables categóricas más importantes para contexto
# AÑADO 'descripcion' AQUÍ

important_cats = ['categoria', 'motivo', 'region', 'tienda', 'negocio', 'linea', 'descripcion']

for cat in important_cats:
    if cat in original_data_for_lookup.columns:
        detailed_results[cat] = original_data_for_lookup[cat].values
    else:
        print(f"Columna '{cat}' no encontrada en original_data_for_lookup.")

# --- MODIFICACIÓN CLAVE AQUÍ PARA AÑADIR LA DESCRIPCIÓN ---
# Generar una columna de descripción combinando variables clave
# AHORA INCLUYE 'descripcion' DEL PRODUCTO
detailed_results['Descripcion_Contextual'] = detailed_results.apply(
    lambda row: f"Merma de '{row.get('descripcion', 'N/A')}' (Cat: {row.get('categoria', 'N/A')}) por {row.get('motivo', 'N/A')} en {row.get('tienda', 'N/A')}.", axis=1
)
# Puedes ajustar esta lambda function para construir la descripción como prefieras
# Por ejemplo, puedes añadir:
# f"Negocio: {row.get('negocio', 'N/A')}, Línea: {row.get('linea', 'N/A')}"
# Asegúrate de que las columnas que usas existan en detailed_results después de la asignación anterior.
# -------------------------------------------------------------

# Ordenar por error para identificar casos problemáticos
detailed_results = detailed_results.sort_values('Error_Absoluto', ascending=False)

# Guardar reporte principal
with open('reporte_prediccion_mermas.md', 'w', encoding='utf-8') as f:
    f.write('# Reporte de Predicción de Mermas\n\n')
    
    f.write('## Resumen Ejecutivo\n\n')
    f.write(f'Se evaluaron {len(trained_models)} modelos de machine learning para predecir {target_variable}. ')
    f.write(f'El mejor modelo fue **{best_model_name}** con un R² de {best_r2:.4f}.\n\n')
    
    f.write('## Comparación de Modelos\n\n')
    f.write('| Modelo | R² | RMSE | MAE | MSE |\n')
    f.write('|--------|----|----- |-----|-----|\n')
    for model_name, metrics in results.items():
        f.write(f"| {model_name} | {metrics['R²']:.4f} | {metrics['RMSE']:.2f} | {metrics['MAE']:.2f} | {metrics['MSE']:.2f} |\n")
    
    f.write(f'\n## Mejor Modelo: {best_model_name}\n\n')
    f.write(f'### Capacidad Predictiva\n')
    f.write(f'- **R²**: {best_r2:.4f} (explica {best_r2*100:.1f}% de la variabilidad)\n')
    f.write(f'- **RMSE**: {best_rmse:.2f} (error promedio de predicción)\n')
    f.write(f'- **Error relativo promedio**: {np.mean(np.abs(relative_errors)):.1f}%\n\n')
    
    f.write('### Implicaciones para Gestión de Mermas\n\n')
    f.write('**Beneficios del modelo:**\n')
    f.write('- Permite anticipar niveles de merma antes de que ocurran\n')
    f.write('- Facilita la planificación de inventarios más precisa\n')
    f.write('- Identifica patrones de merma por categoría, región y motivo\n')
    f.write('- Habilita estrategias preventivas de reducción de pérdidas\n\n')
    
    f.write('**Casos de mayor error (Top 10):**\n\n')
    # RENOMBRAMOS LA COLUMNA A 'Descripción Detallada'
    f.write('| # | Valor Real | Predicción | Error | Error % | Contexto | Descripción Detallada |\n') 
    f.write('|---|------------|------------|-------|---------|----------|-----------------------|\n') # Ajuste para nueva columna
    
    for i, (idx, row) in enumerate(detailed_results.head(10).iterrows()):
        contexto = []
        # Eliminamos 'descripcion' de 'contexto_str' para que no se duplique con 'Descripcion_Contextual'
        # Aquí incluimos solo las categorías principales o de ubicación.
        context_cols_for_table = ['categoria', 'motivo', 'region', 'tienda'] 
        for cat in context_cols_for_table: 
            if cat in row.index: # Asegurarse de que la columna existe en la fila
                contexto.append(f"{cat}: {row[cat]}")
        contexto_str = " | ".join(contexto) # Puedes ajustar el número de elementos aquí
        
        # --- MODIFICACIÓN CLAVE AQUÍ PARA MOSTRAR LA DESCRIPCIÓN ---
        f.write(f"| {i+1} | {row['Valor_Real']:.2f} | {row['Mejor_Prediccion']:.2f} | ")
        f.write(f"{row['Error_Absoluto']:.2f} | {row['Error_Relativo_Pct']:.1f}% | {contexto_str} | {row['Descripcion_Contextual']} |\n")
        # -------------------------------------------------------------
    
    f.write('\n## Recomendaciones\n\n')
    f.write('### Implementación Inmediata\n')
    f.write(f'1. **Usar {best_model_name}** como modelo principal de predicción\n')
    f.write('2. **Monitorear errores** especialmente en casos con error >50%\n')
    f.write('3. **Validar predicciones** con expertos del negocio\n\n')
    
    f.write('### Mejoras Futuras\n')
    f.write('1. **Incorporar variables adicionales**: clima, promociones, días festivos\n')
    f.write('2. **Implementar validación temporal**: entrenar con datos históricos\n')
    ("3. Desarrollar modelos específicos por categoría de producto\n")
    f.write('4. **Establecer alertas automáticas** para predicciones inusuales\n\n')
    
    f.write('*Reporte generado automáticamente por el sistema de análisis predictivo*\n')

print("✓ Reporte principal guardado: reporte_prediccion_mermas.md")

# PASO 15: VISUALIZACIONES
print("\n=== GENERACIÓN DE VISUALIZACIONES ===")

try:
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Gráfico 1: Comparación de modelos
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    models_names = list(results.keys())
    r2_scores = [results[name]['R²'] for name in models_names]
    
    bars = ax.bar(models_names, r2_scores, color=['skyblue', 'lightgreen', 'coral', 'gold'])
    ax.set_ylabel('R² Score')
    ax.set_title('Comparación de Modelos - Capacidad Predictiva (R²)')
    ax.set_ylim(0, max(r2_scores) * 1.1)
    
    # Añadir valores en las barras
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comparacion_modelos_mermas.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: comparacion_modelos_mermas.png")
    
    # Gráfico 2: Predicciones vs Valores Reales (mejor modelo)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(y_test, best_predictions, alpha=0.6, color='blue', s=50)
    
    # Línea diagonal perfecta
    min_val = min(y_test.min(), best_predictions.min())
    max_val = max(y_test.max(), best_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax.set_xlabel(f'{target_variable} - Valores Reales')
    ax.set_ylabel(f'{target_variable} - Predicciones')
    ax.set_title(f'Predicciones vs Valores Reales - {best_model_name}\n(R² = {best_r2:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predicciones_vs_reales_mermas.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: predicciones_vs_reales_mermas.png")
    
    # Gráfico 3: Distribución de errores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma de errores absolutos
    ax1.hist(np.abs(errors), bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax1.set_xlabel('Error Absoluto')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Errores Absolutos')
    ax1.grid(True, alpha=0.3)
    
    # Histograma de errores relativos
    ax2.hist(np.abs(relative_errors), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Error Relativo (%)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Errores Relativos')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribucion_errores_mermas.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: distribucion_errores_mermas.png")
    
except Exception as e:
    print(f"⚠️ Advertencia al generar gráficos: {e}")

# PASO 16: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
print("\n=== ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS ===")

if best_model_name == 'Random Forest' or best_model_name == 'Gradient Boosting':
    try:
        # Obtener modelo entrenado
        best_model = trained_models[best_model_name]
        regressor = best_model.named_steps['regressor']
        
        if hasattr(regressor, 'feature_importances_'):
            # Obtener nombres de características después del preprocesamiento
            preprocessor = best_model.named_steps['preprocessor']
            
            # Características numéricas (mantienen sus nombres)
            num_feature_names = numeric_features
            
            # Características categóricas (se expanden con one-hot encoding)
            if len(categorical_features) > 0:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
                else:
                    # Fallback si get_feature_names_out no está disponible (menos común ahora)
                    # Es una aproximación, puede no ser exacta si hay muchos valores únicos
                    print("⚠️ Advertencia: get_feature_names_out no disponible. Usando nombres de características aproximados para categóricas.")
                    # Generar nombres de características sintéticos, esto es un placeholder
                    # Necesitarías obtener los valores únicos de cada columna categórica para generar nombres más precisos.
                    all_transformed_features = preprocessor.fit_transform(X_train).shape[1]
                    cat_feature_names = [f"cat_feat_{i}" for i in range(all_transformed_features - len(numeric_features))]
            else:
                cat_feature_names = []
            
            # Combinar nombres de características
            all_feature_names = list(num_feature_names) + list(cat_feature_names)
            
            # Obtener importancias
            importances = regressor.feature_importances_
            
            # Crear DataFrame con importancias
            if len(all_feature_names) == len(importances):
                feature_importance_df = pd.DataFrame({
                    'caracteristica': all_feature_names,
                    'importancia': importances
                }).sort_values('importancia', ascending=False)
                
                print("✓ Top 15 características más importantes:")
                print(feature_importance_df.head(15))
                
                # Guardar análisis de importancia
                with open('importancia_caracteristicas_mermas.md', 'w', encoding='utf-8') as f:
                    f.write('# Análisis de Importancia de Características\n\n')
                    f.write(f'## Modelo: {best_model_name}\n\n')
                    f.write('### Top 15 Características Más Importantes\n\n')
                    f.write('| Ranking | Característica | Importancia | Interpretación |\n')
                    f.write('|---------|----------------|-------------|----------------|\n')
                    
                    for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
                        interpretacion = "Variable numérica" if row['caracteristica'] in numeric_features else "Variable categórica"
                        f.write(f"| {i} | {row['caracteristica']} | {row['importancia']:.4f} | {interpretacion} |\n")
                    
                    f.write('\n### Insights para Gestión de Mermas\n\n')
                    f.write('Las características más importantes identificadas por el modelo indican:\n\n')
                    
                    top_features = feature_importance_df.head(5)['caracteristica'].tolist()
                    for i, feature in enumerate(top_features, 1):
                        # Intentar una interpretación más detallada si la característica es una categoría o un one-hot encoding
                        if any(cat_col in feature for cat_col in categorical_features):
                            original_cat_col = next((cat_col for cat_col in categorical_features if cat_col in feature), "Categoría")
                            f.write(f'{i}. **{feature}**: Una de las categorías importantes de **{original_cat_col}** que influye significativamente en la merma.\n')
                        elif any(temp in feature.lower() for temp in ['mes', 'año', 'trimestre', 'dia_semana']):
                            f.write(f'{i}. **{feature}**: Factor temporal significativo que puede indicar estacionalidad o patrones diarios/mensuales de merma.\n')
                        else:
                            f.write(f'{i}. **{feature}**: Factor operacional importante que requiere atención para reducir mermas.\n')
                    
                    f.write('\n*Este análisis ayuda a enfocar esfuerzos de reducción de mermas en los factores más influyentes.*\n')
                
                print("✓ Análisis de importancia guardado: importancia_caracteristicas_mermas.md")
                
            else:
                print("⚠️ No se pudo generar análisis de importancia por diferencias dimensionales entre las características y las importancias.")
                print(f"Número de nombres de características esperados: {len(all_feature_names)}")
                print(f"Número de importancias obtenidas: {len(importances)}")
                
        else:
            print(f"⚠️ El modelo {best_model_name} no tiene el atributo 'feature_importances_'.")
            
    except Exception as e:
        print(f"⚠️ Error en análisis de importancia: {e}")

# PASO 17: RESUMEN FINAL Y RECOMENDACIONES
print("\n" + "="*70)
print("🎯 RESUMEN FINAL - ANÁLISIS PREDICTIVO DE MERMAS")
print("="*70)

print(f"\n📊 MEJORES RESULTADOS:")
print(f"• Modelo óptimo: {best_model_name}")
print(f"• Capacidad predictiva: {best_r2:.1%} de la variabilidad explicada")
print(f"• Error promedio: ±{best_rmse:.2f} unidades de {target_variable}")
print(f"• Total de registros analizados: {len(X)}")

print(f"\n💡 APLICACIÓN PRÁCTICA:")
print("1. 🔮 Predicción proactiva de mermas futuras")
print("2. 📈 Optimización de niveles de inventario")
print("3. 🎯 Identificación de productos/ubicaciones de alto riesgo")
print("4. 💰 Estimación de impacto económico de pérdidas")

print(f"\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
print("1. Implementar el modelo en ambiente de producción")
print("2. Establecer monitoreo continuo de performance")
print("3. Desarrollar dashboards de alertas tempranas")
print("4. Capacitar equipos en interpretación de resultados")
print("5. Iterar y mejorar el modelo con nuevos datos")

print(f"\n📁 ARCHIVOS GENERADOS:")
print("• reporte_prediccion_mermas.md (reporte principal)")
print("• comparacion_modelos_mermas.png (comparación visual)")
print("• predicciones_vs_reales_mermas.png (validación del modelo)")
print("• distribucion_errores_mermas.png (análisis de errores)")
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("• importancia_caracteristicas_mermas.md (factores clave)")

print(f"\n✅ ANÁLISIS PREDICTIVO DE MERMAS COMPLETADO EXITOSAMENTE")
print("="*70)

# Nota final sobre adaptaciones específicas
print(f"\n📋 NOTA SOBRE ADAPTACIONES REALIZADAS:")
print("• ✓ Cambio de objetivo: de maximización de ventas a minimización de mermas")
print("• ✓ Variables adaptadas: uso de campos específicos del dataset de mermas")
print("• ✓ Modelo adicional: Gradient Boosting para mejor capacidad predictiva")
print("• ✓ Métricas contextualizadas: interpretación desde perspectiva de pérdidas")
print("• ✓ Análisis de impacto: consideración del costo económico de errores")
print("• ✓ Descripción contextual añadida en el reporte de errores para mayor claridad, incluyendo la descripción del producto.")
